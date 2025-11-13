# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 21:26:35 2025

@author: gauthambekal93
"""

import os
import sys
import json
import argparse
import torch
import pickle
import torchvision
from torchvision import datasets, transforms



def image_net(arguments):
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment", type=str, default='cfg/a.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)
    
    
    source_train_data_dir = params["source_train_data_dir"]
    source_val_data_dir = params["source_val_data_dir"]
 
    #normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    
    transform = transforms.Compose( [transforms.Resize((32, 32)), transforms.ToTensor(),])
    
    trainset = datasets.ImageFolder(root = os.path.join(project_root, source_train_data_dir), transform=transform)
    testset = datasets.ImageFolder(root = os.path.join(project_root , source_val_data_dir ),  transform=transform)
    class_to_idx = trainset.class_to_idx

    train_x, train_y = [], []
    
    for i, (image, label) in enumerate(trainset):
        print(i)
        train_x.append( image)
        train_y.append( label ) 
        
    
    train_x = torch.stack(train_x, dim = 0 )
    train_y = torch.tensor(train_y, dtype = torch.long  )
    
    val_x, val_y = [], []
    
    for i, (image, label) in enumerate(testset):
        print(i)
        val_x.append( image)
        val_y.append( label )

    
    val_x = torch.stack(val_x, dim = 0 )
    val_y = torch.tensor(val_y, dtype = torch.long )
     
    with open(os.path.join(project_root, params['dest_data_dir']), 'wb+') as f:
        pickle.dump([train_x, train_y, class_to_idx, val_x, val_y], f)




if __name__ == '__main__':
    """
    Generates all the required data
    """
    
    project_root = os.path.abspath ( os.path.join( os.getcwd(), "..","..") )
    
    confguration_path = os.path.join(project_root,"configuration_files","split_image_net","data", "0.json")
    
    sys.exit(image_net( ['-c',  confguration_path  ] ) )# we use the hyperparameters stored in env_temp_cfg to create data for a specifc run




