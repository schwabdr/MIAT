import sys

import numpy as np
import torch.utils.data as Data
from PIL import Image

# import tools
import torch
import util


class data_noise_dataset(Data.Dataset):
    def __init__(self, img_path, noisy_label_path, clean_label_path):
        
        self.train_data = np.load(img_path).astype(np.float32) # B C H W # what is B? batches?, Channel, Height, Width
        # I'm going to intuit that B stands for the index of the image. 
        self.train_noisy_labels = np.load(noisy_label_path)
        self.train_clean_labels = np.load(clean_label_path)

    def __getitem__(self, index):
        img, noisy_label, clean_label = self.train_data[index], self.train_noisy_labels[index], \
                                        self.train_clean_labels[index]

        img = torch.from_numpy(img)
        noisy_label = torch.from_numpy(np.array(noisy_label)).long()
        clean_label = torch.from_numpy(np.array(clean_label)).long()
     
        return img, noisy_label, clean_label, index

    def __len__(self):

        return len(self.train_data)


class data_dataset(Data.Dataset):
    def __init__(self, img_path, clean_label_path, transform=None):
        self.transform = transform
        self.train_data = np.load(img_path)
        self.train_clean_labels = np.load(clean_label_path).astype(np.float32)
        self.train_clean_labels = torch.from_numpy(self.train_clean_labels).long()
    #this function below does return the correct labels - I'm not sure if the image is correct.
    #still not sure I've put the data in the correct format on disk to be read in.
    def __getitem__(self, index):
        #print("__getitem__() inside data.py:")
        img, clean_label = self.train_data[index], self.train_clean_labels[index]
        #print("img (before fromarray): ", img)
        img = Image.fromarray(img)
        #print("img (after fromarray): ", img)
        if self.transform is not None:
            img = self.transform(img)
        #print("img (after transform): ", img)
        
        #util.print_tensor_details("img", img)
        #print(f"img: {img}")
        #util.print_tensor_details("clean_label", clean_label)
        #print(f"clean_label: {clean_label}")
        return img, clean_label

    def __len__(self):
        return len(self.train_data)


class distilled_dataset(Data.Dataset):
        def __init__(self, distilled_images, distilled_noisy_labels, distilled_bayes_labels):
            
            self.distilled_images = distilled_images
            self.distilled_noisy_labels = distilled_noisy_labels
            self.distilled_bayes_labels = distilled_bayes_labels 
            # print(self.distilled_images)

        def __getitem__(self, index):
            # print(index)
            img, bayes_label, noisy_label = self.distilled_images[index], self.distilled_bayes_labels[index], self.distilled_noisy_labels[index]
            # print(img)
            # print(bayes_label)
            # print(noisy_label)

            img = torch.from_numpy(img)
            bayes_label = torch.from_numpy(np.array(bayes_label)).long()
            noisy_label = torch.from_numpy(np.array(noisy_label)).long()

            return img, bayes_label, noisy_label, index
        
        def __len__(self):
            return len(self.distilled_images)
    

