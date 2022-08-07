from utils import dataload
from utils import utils
import data
import numpy as np
import os
import argparse
import config

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

print(f"hello MIAT")

data_dict = dataload.unpickle("./data/cifar-10-batches-py/data_batch_1")
#label data - names are in this file
label_dict = dataload.unpickle_string("./data/cifar-10-batches-py/batches.meta")

X = data_dict[b'data'] #the b has something to do with binary encoding of string ... not sure - it works.
Y = data_dict[b'labels']

data_dict_test = dataload.unpickle("./data/cifar-10-batches-py/test_batch")
X_test = data_dict[b'data']
Y_test = data_dict[b'labels']

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8") #reshape makes X into numpy array
Y = np.array(Y).astype("uint8") #do same for Y, no reshape needed

X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8") #reshape makes X into numpy array
Y_test = np.array(Y_test).astype("uint8") #do same for Y, no reshape needed

#index into classes for the correct label
classes = label_dict['label_names']  #['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#utils.displayRandomImgGrid(X, Y, classes, rows=5, cols=5, Y_hat=None)


c = config.configuration()
args = c.getArgs()


#https://numpy.org/doc/stable/reference/generated/numpy.save.html

print("Saving numpy arrays to file ...")
with open(args.nat_img_train, 'wb') as f:
    np.save(f, X)
with open(args.nat_label_train, 'wb') as f:
    np.save(f, Y)
with open(args.nat_img_test, 'wb') as f:
    np.save(f,X_test)
with open(args.nat_label_test, 'wb') as f:
    np.save(f,Y_test)
print("Save complete!")


print(f"Hello again MIAT, EOF")
