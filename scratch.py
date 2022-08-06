#The purpose of this file is to hold scratch work to keep my "actual" code clean
#For instance, I did some work with numpy arrays and reshape on the CIFAR10 data - I want to keep that code around
#but I don't want it in my "initialize.py" file - so I'll move it here.

from utils import dataload
import data
import numpy as np
import os
import argparse

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

print(f"hello MIAT")

data_dict = dataload.unpickle("./data/cifar-10-batches-py/data_batch_1")

#print(data_dict)

print("keys:", data_dict.keys())

data1 = data_dict[b'data']
#print(data1)
print(data1[0][0])
print(data1[0][1])
print(data1[0][2])

print(data1[0][3069])
print(data1[0][3070])
print(data1[0][3071])


print(f"Dim: {data1.ndim}, Shape: {data1.shape}, Size: {data1.size}, Len: {len(data1)}")
# goal from data.py # B C H W
data2 = data1.reshape(10000, 3, 32, 32)

print(f"Dim: {data2.ndim}, Shape: {data2.shape}, Size: {data2.size}, Len: {len(data2)}")

#print(data2)
print(data2[0][0][0][0])
print(data2[0][0][0][1])
print(data2[0][0][0][2])

print(data2[0][2][31][29])
print(data2[0][2][31][30])
print(data2[0][2][31][31])

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
args = parser.parse_args()

# a note here - since "-" is an operator, apparently the "--nat-img-train" from above gets converted to underscore here
#print(args.nat_img_train)

x = np.arange(120) + 1
x = x.reshape(2,60) #this is a proxy for my data array above
x = x.reshape(2,3,4,5) # a 3x4 3 rows, 4 columns, 3 ht, 4 width"img" with 3 channels

#print(x)

# print types of variables
#print("type(X):", type(X))
#print("type(Y):", type(Y))
