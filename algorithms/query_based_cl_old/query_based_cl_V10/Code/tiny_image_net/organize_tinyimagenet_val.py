# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 08:01:21 2026

@author: gauthambekal93
"""

"""
You can see that in the train folder, all images are already placed inside class-named subfolders, just like ImageNet — so you don’t need to change that.
However, the val folder also needs to be reorganized into class-named subfolders (like ImageNet) in order for PyTorch’s ImageFolder to read it correctly.
We can achieve this using the following script:
 
"""
import glob
import os
from shutil import move
from os import rmdir
import torch
from torchvision import datasets, transforms

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent   # go up two levels, adjust as needed

os.chdir(ROOT)

target_folder = 'datasets/tiny-imagenet-200/val/'

val_dict = {}
with open('datasets/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('	')
        val_dict[split_line[0]] = split_line[1]
        
paths = glob.glob('datasets/tiny-imagenet-200/val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')
       
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/images/' + str(file)
    move(path, dest)
    
rmdir('datasets/tiny-imagenet-200/val/images')

#This makes Tiny-ImageNet’s file format basically the same as ImageNet’s. You can now load it using similar PyTorch DataLoader code. In the following code, we resize the images to 32×32 for processing.

def tiny_loader( data_dir):
    num_label = 200
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         normalize, ])
    transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
    #train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    #test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return trainset, testset, num_label


data_dir = "datasets/tiny-imagenet-200"

trainset, testset, num_label = tiny_loader( data_dir)
            
            
            
            