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
#label data
label_dict = dataload.unpickle_string("./data/cifar-10-batches-py/batches.meta")


X = data_dict[b'data'] #the b has something to do with binary encoding of string ... not sure - it works.
Y = data_dict[b'labels']

X = X.reshape(10000, 3, 32, 32) #reshape makes X into numpy array
Y = np.array(Y) #do same for Y, no reshape needed
#index into classes for the correct label
classes = label_dict['label_names']  #['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#print("classes:",classes)

#Now lets show some images.
# thank you stack exchange
#https://stackoverflow.com/questions/35995999/why-cifar-10-images-are-not-displayed-properly-using-matplotlib
#https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image
#X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
#For adding labels see this: https://stackoverflow.com/questions/42435446/how-to-put-text-outside-python-plots
#https://stackoverflow.com/questions/61341119/write-a-text-inside-a-subplot
X_v = X.transpose(0,2,3,1).astype("uint8")

#Visualizing CIFAR 10
fig, axes1 = plt.subplots(5,5,figsize=(5,5))
for j in range(5):
    for k in range(5):
        i = np.random.choice(range(len(X_v)))
        axes1[j][k].set_axis_off()
        axes1[j][k].imshow(X_v[i:i+1][0],interpolation='nearest')
        axes1[j][k].text(0,0,classes[Y[i]]) # this gets the point accross but needs fixing.
plt.show()


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--nat-img-train', type=str, help='natural training data', default='./data/train_images.npy')
args = parser.parse_args()

# a note here - since "-" is an operator, apparently the "--nat-img-train" from above gets converted to underscore here
#print(args.nat_img_train)
