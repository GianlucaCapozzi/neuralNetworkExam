import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


train = {"acc":[], "err":[], "loss":[]}
test = {"acc":[], "err":[], "loss":[]}
pgd = {"acc":[], "err":[], "loss":[]}

train2 = {"acc":[], "err":[], "loss":[]}
test2 = {"acc":[], "err":[], "loss":[]}
pgd2 = {"acc":[], "err":[], "loss":[]}

times = []

#[0] -> acc, [1] -> pgd
max_test = [-1.,-1.,""]
max_pgd =  [-1.,-1.,""]
last_epoch = ""



PATH1 = "output/new_train_free_output_ResNet50_m8_e90"
FILENAME1 = PATH1 + "/log.txt"

PATH2 = "output/new_train_free_output_PreActResNet18_m8_e90"
FILENAME2 = PATH2 + "/log.txt"

with open(FILENAME1,"r") as f:
    for line in f:

        if(line.split(" ")[1] == "Epoch:"):
            last_epoch = str(line.split(" ")[2] )

        if(line.split(" ")[0] == "Test"):
            values = line.split(":")[1].split(",")
            test["acc"].append( float(values[0]))
            test["err"].append( float(values[1]))
            test["loss"].append( float(values[2]))
        elif(line.split(" ")[0] == "PGD" or line.split(" ")[0] == "PGD_50"):
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

# print(train,test)
# print("MAX TEST ACC",max_test)
# print("MAX PGD ACC",max_pgd)
# print("AVG TIME", np.average(times))


with open(FILENAME2,"r") as f:
    for line in f:

        if(line.split(" ")[1] == "Epoch:"):
            last_epoch = str(line.split(" ")[2] )

        if(line.split(" ")[0] == "Test"):
            values = line.split(":")[1].split(",")
            test2["acc"].append( float(values[0]))
            test2["err"].append( float(values[1]))
            test2["loss"].append( float(values[2]))
        elif(line.split(" ")[0] == "PGD" or line.split(" ")[0] == "PGD_50"):
            values = line.split(":")[1].split(",")
            pgd2["acc"].append( float(values[0]))
            pgd2["err"].append( float(values[1]))
            pgd2["loss"].append( float(values[2]))
  
    
        elif(line.split(" ")[0] == "Epoch"):
            value = line.split(":")[1].split(" ")[1]      
            times.append(float(value))


        elif(line.split(",")[0]=="Train acc" ):
            values = line.split(":")[1].split(",")
            train2["acc"].append( float(values[0]))
            train2["err"].append( float(values[1]))
            train2["loss"].append( float(values[2]))
        
        if(test["acc"]!=[]  and pgd["acc"]!=[] and test["acc"][-1] > max_test[0]):
            max_test[0] = test["acc"][-1]
            max_test[1] = pgd["acc"][-1]
            max_test[2] = last_epoch 
        if( test["acc"]!=[] and pgd["acc"]!=[] and  pgd["acc"][-1] > max_pgd[1]):
            max_pgd[0] = test["acc"][-1]
            max_pgd[1] = pgd["acc"][-1]
            max_pgd[2] = last_epoch 



# /***************************************
#  *                  #                  *
#  *                  #                  *
#  * -----------TRAIN------------------- *
#  *                  #                  *
#  *                  #                  *
#  *                  #                  *
#  ***************************************/



# /********
#  * ACC *
#  ********/

acc_val =  test["acc"]
acc_val2 = test2["acc"]

epochs = range(1,len(test["acc"])+1)

plt.plot(epochs, acc_val, c='b', label='ResNet50 Validation Acc.')
plt.plot(epochs, acc_val2, c='g', label='PreAct18 Validation Acc.')
plt.title('Free ResNet50 vs Free PreActResNet18 Validation Acc.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




# /********
#  * ACC *
#  ********/

acc_val = pgd["acc"]
acc_val2 = pgd2["acc"]

epochs = range(1,len(pgd["acc"])+1)

plt.plot(epochs, acc_val, c='b', label='ResNet50 PGD-50 Acc.')
plt.plot(epochs, acc_val2, c='g', label='PreAct18 PGD-50 Acc.')
plt.title('Free ResNet50 vs Free PreActResNet18 PGD-50 Acc.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



