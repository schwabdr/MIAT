import numpy as np

import torch
import torchvision

import torchvision.transforms as transforms

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# size(imgs, channels, height, width)
x = np.random.randint(low=0, high=255,size=(1,3,10,15),dtype=np.int)

print(x)

normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
#https://github.com/kuangliu/pytorch-cifar/issues/19
#stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
#normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))


x = x / 255

print(x)

x_tensor = torch.tensor(x)

print(x_tensor)

x_tensor_normalized = normalize(x_tensor[0])

print(x_tensor_normalized)
print("min (norm): ", torch.min(x_tensor_normalized)) #I can't figure why the min = -2.4291 and max = 2.7537
print("max (norm): ", torch.max(x_tensor_normalized))

X_v = x_tensor.detach().cpu().numpy().transpose(0,2,3,1)
X_vn = x_tensor_normalized.detach().cpu().numpy().reshape(1,3,10,15).transpose(0,2,3,1)

rows = 1
cols = 2
fig, axes1 = plt.subplots(rows,cols,figsize=(5,5))
axes1[0].set_axis_off()
axes1[1].set_axis_off()
axes1[0].imshow(X_v[0],interpolation='nearest')
axes1[1].imshow(X_vn[0],interpolation='nearest')
plt.show()

        