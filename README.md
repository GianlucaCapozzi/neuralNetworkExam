# neuralNetworkExam
Work for Neural Network course  2020 Sapienza University


## File system
1. CIFAR10: folder containing Working code
2. paper: folder containing source papers 
3. pdf: folder containing the report
In CIFAR10/output folder are contained some of the several experiments results, the most significant are presented in the paper


## Requirements: 
Python 3.6 

Install the required packages:
```
$ pip install -r requirements.txt
```

Follow the instructions below to install apex:
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Usage
In configs.ini change settings for comparison.
```
$ python3 main_script.py
```