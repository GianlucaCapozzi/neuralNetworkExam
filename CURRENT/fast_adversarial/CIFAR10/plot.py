import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


train = {"acc":[], "err":[], "loss":[]}
test = {"acc":[], "err":[], "loss":[]}
pgd = {"acc":[], "err":[], "loss":[]}
times = []

#[0] -> acc, [1] -> pgd
max_test = [-1.,-1.,""]
max_pgd =  [-1.,-1.,""]
last_epoch = ""

PATH = "output/new_train_fast_output_PreActResNet18"
FILENAME = PATH + "/log.txt"

with open(FILENAME,"r") as f:
    for line in f:

        if(line.split(" ")[1] == "Epoch:"):
            last_epoch = str(line.split(" ")[2] )

        if(line.split(" ")[0] == "Test"):
            values = line.split(":")[1].split(",")
            test["acc"].append( float(values[0]))
            test["err"].append( float(values[1]))
            test["loss"].append( float(values[2]))
        elif(line.split(" ")[0] == "PGD"):
            values = line.split(":")[1].split(",")
            pgd["acc"].append( float(values[0]))
            pgd["err"].append( float(values[1]))
            pgd["loss"].append( float(values[2]))
  
    
        elif(line.split(" ")[0] == "Epoch"):
            value = line.split(":")[1].split(" ")[1]      
            times.append(float(value))


        elif(line.split(",")[0]=="Train acc" ):
            values = line.split(":")[1].split(",")
            train["acc"].append( float(values[0]))
            train["err"].append( float(values[1]))
            train["loss"].append( float(values[2]))
        
        if(test["acc"]!=[]  and pgd["acc"]!=[] and test["acc"][-1] > max_test[0]):
            max_test[0] = test["acc"][-1]
            max_test[1] = pgd["acc"][-1]
            max_test[2] = last_epoch 
        if( test["acc"]!=[] and pgd["acc"]!=[] and  pgd["acc"][-1] > max_pgd[1]):
            max_pgd[0] = test["acc"][-1]
            max_pgd[1] = pgd["acc"][-1]
            max_pgd[2] = last_epoch 

print(train,test)
print("MAX TEST ACC",max_test)
print("MAX PGD ACC",max_pgd)
print("AVG TIME", np.average(times))

"""
Train acc, err, loss: 0.289, 0.711, 1.909
Epoch time: 1.8350 minutes
Test acc, err, loss: 0.377, 0.623, 1.668
PGD acc: 0.377, 0.623, 1.668
"""

# /***************************************
#  *                  #                  *
#  *                  #                  *
#  * -----------TRAIN------------------- *
#  *                  #                  *
#  *                  #                  *
#  *                  #                  *
#  ***************************************/



# /********
#  * LOSS *
#  ********/

loss_train = train["loss"]
loss_val = test["loss"]
epochs = range(1,len(train["loss"])+1)


plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# /********
#  * ACC *
#  ********/

acc_train = train["acc"]
acc_val = test["acc"]
epochs = range(1,len(train["acc"])+1)




plt.plot(epochs, acc_train, 'g', label='Training acc')
plt.plot(epochs, acc_val, 'b', label='validation acc')
plt.title('Training and Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# /********
#  * ERR *
#  ********/



err_train = train["err"]
err_val = test["err"]
epochs = range(1,len(train["err"])+1)




plt.plot(epochs, err_train, 'g', label='Training err')
plt.plot(epochs, err_val, 'b', label='validation err')
plt.title('Training and Validation err')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()










# /***************************************
#  *                  #                  *
#  *                  #                  *
#  * -----------PGD------------------- *
#  *                  #                  *
#  *                  #                  *
#  *                  #                  *
#  ***************************************/



# /********
#  * LOSS *
#  ********/

loss_pgd = pgd["loss"]
epochs = range(1,len(pgd["loss"])+1)

plt.plot(epochs, loss_pgd, 'r', label='Test pgd')
plt.title('PGD Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# /********
#  * ACC *
#  ********/

acc_pgd = pgd["acc"]
epochs = range(1,len(pgd["acc"])+1)

plt.plot(epochs, acc_pgd, 'r', label='Test pgd')
plt.title('PGD Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()



# /********
#  * ERR *
#  ********/

err_pgd = pgd["err"]
epochs = range(1,len(pgd["err"])+1)

plt.plot(epochs, err_pgd, 'r', label='Test pgd')
plt.title('PGD Validation err')
plt.xlabel('Epochs')
plt.ylabel('Err')
plt.legend()
plt.show()






