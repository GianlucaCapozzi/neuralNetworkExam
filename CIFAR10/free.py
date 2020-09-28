import argparse
import logging
import os
import time
import sys

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision.models as models
from preact_resnet import PreActResNet18
from utils import *

# IMPORT WIDE-RESNET
from wideresnet import *

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../data', type=str)
    parser.add_argument('--epochs', default=90, type=int, help='Total number of epochs will be this argument * number of minibatch replays.')
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.04, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--model-name', default="PreActResNet18", type=str)
    parser.add_argument('--minibatch-replays', default=8, type=int)
    parser.add_argument('--out-dir', default='train_free_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--custom-name', default="asd", type=str)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()

args = get_args()
logger = initiate_logger("new_" + args.out_dir + "_PreActResNet18_"+args.custom_name)
print = logger.info
cudnn.benchmark = True

# Check if there is cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")


def main():

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, 2, 32)

    epsilon = (args.epsilon / 255.) / std

    #model = models.resnet50().to(device)
    model = PreActResNet18().to(device)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    delta.requires_grad = True

    lr_steps = args.epochs * len(train_loader) * args.minibatch_replays
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    total_time = 0

    for epoch in range(args.epochs):

        start_train = time.time()

        train(train_loader, model, args.minibatch_replays, criterion, epoch, epsilon, delta, opt, scheduler)
        
        end_train = time.time()

        epoch_time = (end_train - start_train)/60
        total_time += epoch_time

        print("Epoch time: %.4f minutes", epoch_time)

        # Evaluation
        best_state_dict = model.state_dict()
        #model_test = models.resnet50().cuda()
        model_test = PreActResNet18().to(device)
        model_test.load_state_dict(best_state_dict)
        model_test.float()
        model_test.eval()

        # Evaluate standard acc on test set
        test_loss, test_acc, test_err = evaluate_standard(test_loader, model_test)
        print("Test acc, err, loss: %.3f, %.3f, %.3f" %(test_acc, test_err, test_loss))

        '''
        # Evaluate acc against PGD_10 attack
        pgd_loss, pgd_acc, pgd_err = evaluate_pgd(test_loader, model_test, 10, 1)
        print("PGD_10 acc, err, loss: %.3f, %.3f, %.3f" %(pgd_acc, pgd_err, pgd_loss))

        # Evaluate acc against PGD_20 attack
        pgd_loss, pgd_acc, pgd_err = evaluate_pgd(test_loader, model_test, 20, 1)
        print("PGD_20 acc, err, loss: %.3f, %.3f, %.3f" %(pgd_acc, pgd_err, pgd_loss))
        '''

        # Evaluate acc against PGD_50 attack
        pgd_loss, pgd_acc, pgd_err = evaluate_pgd(test_loader, model_test, 50, 1)
        print("PGD_50 acc, err, loss: %.3f, %.3f, %.3f" %(pgd_acc, pgd_err, pgd_loss))
    
    logger.info('Total train time: %.4f minutes', total_time)


def train(train_loader, model, replays, criterion, epoch, epsilon, delta, opt, scheduler):
    start_epoch_time = time.time()
    train_loss = 0
    train_acc = 0
    train_n = 0
    train_err = 0

    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (X, y) in enumerate(train_loader):
        end = time.time()
        X, y = X.cuda(), y.cuda()
        for _ in range(args.minibatch_replays):
            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)

            prec1, prec5 = accuracy(output, y, topk=(1, 5))
            losses.update(loss.item(), X.size(0))
            top1.update(prec1[0], X.size(0))
            top5.update(prec5[0], X.size(0))

            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + epsilon * torch.sign(grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            opt.step()
            delta.grad.zero_()
            scheduler.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Train Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, top1=top1, top5=top5,cls_loss=losses))
                sys.stdout.flush()

        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_err += (output.max(1)[1] != y).sum().item()
        train_n += y.size(0)

    lr = scheduler.get_lr()[0]

    print("Train acc, err, loss: %.3f, %.3f, %.3f" %(train_acc / len(train_loader.dataset), train_err / len(train_loader.dataset), train_loss / len(train_loader.dataset)))

if __name__ == "__main__":
    main()
