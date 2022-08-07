# MISC utils in this file - such as displaying images
import numpy as np

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

#displays a grid of images with labels.
#we assume input is in shape B, C, H, W (img count, channels, height, width)
#assume X is NOT normalized - caller must ensure this for now.
def displayRandomImgGrid(X, Y, classes, rows=5, cols=5, Y_hat=None):
    #Now lets show some images.
    # thank you stack exchange
    #https://stackoverflow.com/questions/35995999/why-cifar-10-images-are-not-displayed-properly-using-matplotlib
    #https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image
    #X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    #For adding labels see this: https://stackoverflow.com/questions/42435446/how-to-put-text-outside-python-plots
    #https://stackoverflow.com/questions/61341119/write-a-text-inside-a-subplot
    X_v = X.transpose(0,2,3,1).astype("uint8") 

    #Visualizing CIFAR 10
    fig, axes1 = plt.subplots(rows,cols,figsize=(5,5))
    for j in range(rows):
        for k in range(cols):
            i = np.random.choice(range(len(X_v)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X_v[i:i+1][0],interpolation='nearest')
            axes1[j][k].text(0,0,classes[Y[i]]) # this gets the point accross but needs fixing.
            if Y_hat is not None:
                print("not implemented yet") #add the predicted label
    plt.show()
    