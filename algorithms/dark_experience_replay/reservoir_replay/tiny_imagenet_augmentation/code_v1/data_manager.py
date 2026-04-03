# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:09:43 2025

@author: gauthambekal93
"""


import os
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from queue import Queue
import time
from augmentations import get_task_params, augment_batch
from torch.utils.data import DataLoader

class DataManager:
     
     def __init__(self, device, root, data_dir, classes_per_task, total_classes, num_old_task_window, num_datapoints_per_timestep, num_tasks, buffer_size, samples_from_buffer):    
       
         self.device = device
         self.classes_per_task = classes_per_task
    
         self.total_classes = total_classes
         self.current_task_id = 0
         self.num_old_task_window = num_old_task_window
         
         self.num_tasks = num_tasks
         
         self.data_path = os.path.join( root, data_dir)
         
         self.pad = 4 
        
         self.train_x, self.train_y,  self.test_x, self.test_y  = [] , [] , [], [] 
         
         self.task_train_x, self.task_train_y ,self.task_test_x, self.task_test_y = {}, {}, {}, {}
         
         self.buffer_x = torch.empty(buffer_size, 3, 32, 32).to(self.device) 
         self.buffer_z = torch.empty(buffer_size, classes_per_task).to(self.device)
         self.buffer_y = torch.empty(buffer_size).to(self.device).long() 
         
         self.step = 0
         self.buffer_counter = 0
         
         self.buffer_size = buffer_size
         self.samples_from_buffer = samples_from_buffer
         

         
     def create_tiny_imagenet_data(self):
      
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        
        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             normalize, ])
        transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
        
        self.train_set = datasets.ImageFolder(root=os.path.join(self.data_path, 'train'), transform=transform_train)
        
        self.test_set = datasets.ImageFolder(root=os.path.join(self.data_path, 'val'), transform=transform_test)

        
        assert self.train_set.class_to_idx == self.test_set.class_to_idx, "DATA LOADING WENT WRONG"
       
        loader = DataLoader(self.train_set, batch_size=500, shuffle=False,num_workers=1, pin_memory=True )
        

        train_count =  int(500 * 0.80)
        
        for img, label in loader:
            rand_idx = torch.randperm( 500 )
            train_idx =  rand_idx[:train_count]
            val_idx =  rand_idx[train_count:]
            
            self.train_x.append(img[train_idx])
            self.train_y.append(label[train_idx])
            
            self.val_x.append(img[val_idx])
            self.val_y.append(label[val_idx])
            
        
            
        self.train_x = torch.cat (self.train_x, dim =0)
        
        self.train_y = torch.cat(self.train_y, dim =0)
        
        self.val_x = torch.cat (self.val_x, dim =0)
        
        self.val_y = torch.cat(self.val_y, dim =0)
        
        loader = DataLoader(self.test_set, batch_size=1000, shuffle=False,num_workers=1, pin_memory=True )
        
        count =0
        for img, label in loader:
            print(count)
            self.test_x.append(img)
            self.test_y.append(label)
            count +=1
            
            
        self.test_x =torch.cat (self.test_x, dim =0)
        
        self.test_y = torch.cat(self.test_y, dim =0)
        
        #save_path = os.path.join(self.data_path, "tiny_imagenet_cached.pt")
        
        save_path = os.path.join(self.data_path, "tiny_imagenet_cached-v2.pt")
        
        torch.save(
            {
                "train_x": self.train_x.cpu(),
                "train_y": self.train_y.cpu(),
                "val_x": self.val_x.cpu(),
                "val_y": self.val_y.cpu(),
                "test_x":  self.test_x.cpu(),
                "test_y":  self.test_y.cpu(),
                "class_to_idx": self.train_set.class_to_idx,
            },
            save_path
        )
        
        print(f"Saved Tiny ImageNet cache to {save_path}")

            
     def load_tiny_imagenet_data(self):
         
         load_path = os.path.join(self.data_path, "tiny_imagenet_cached-v2.pt")

         data = torch.load(load_path, map_location="cpu")
        
         self.train_x = data["train_x"].to(self.device)
         self.train_y = data["train_y"].to(self.device)
         
         self.val_x = data["val_x"].to(self.device)
         self.val_y = data["val_y"].to(self.device)
         
         self.test_x  = data["test_x"].to(self.device)
         self.test_y  = data["test_y"].to(self.device)
        
         self.class_to_idx = data["class_to_idx"]
        
         print("Loaded Tiny ImageNet cache")
        
        

     
     def relable_data(self, Y, task_labels):
         
            Y_new = torch.empty_like(Y)
        
            for new_label, old_label in enumerate(task_labels):
                Y_new[Y == old_label] = new_label
        
            return Y_new
                 
                  
     def create_task_data(self):
            
             task_labels = torch.randperm(self.total_classes)[: self.classes_per_task ].to(self.device)
             
             mask = torch.isin( self.train_y,  task_labels)
             
             train_x, train_y = self.train_x[mask], self.train_y[mask]
             
             #pixel_permutation = torch.randperm(train_x.shape[2], device=self.device)
            
             data_permutation = torch.randperm(train_x.shape[0], device=self.device)
             
             params = get_task_params(self.current_task_id)
             
             train_x = augment_batch(train_x, params)

             #train_x = train_x[:, :, pixel_permutation, :]
             
             #train_x = train_x[:, :, :, pixel_permutation]
             
             train_x = train_x[data_permutation]
             
             train_y = train_y[data_permutation]
             
             self.task_train_x[self.current_task_id] = train_x
            
             self.task_train_y[self.current_task_id] = self.relable_data(train_y, task_labels)
             
             
             mask = torch.isin( self.test_y,  task_labels)
             
             test_x , test_y =  self.test_x[mask], self.test_y[mask]
             
             #test_x = test_x[:, :, pixel_permutation, :]
             
             #test_x = test_x[:, :, :, pixel_permutation]
             
             test_x = augment_batch(test_x, params)
             
             self.task_test_x[self.current_task_id] = test_x
            
             self.task_test_y[self.current_task_id] = self.relable_data(test_y, task_labels) 

        

     def fill_buffer(self, x, z, y):
         
         z = z.detach()
         B = x.size(0)
         for i in range(B):
             self.step += 1
     
             if self.buffer_counter < self.buffer_size:
                 # fill phase
                 self.buffer_x[self.buffer_counter].copy_(x[i].clone())
                 self.buffer_y[self.buffer_counter] = y[i].clone()
                 self.buffer_z [self.buffer_counter].copy_(z[i].clone())
                 self.buffer_counter += 1
             else:
                 # reservoir step
                 j = torch.randint(0, self.step, () ).item()  
                 if j < self.buffer_size:
                     self.buffer_x[j].copy_(x[i].clone())
                     self.buffer_z[j].copy_(z[i].clone())
                     self.buffer_y[j] = y[i].clone()                     
                     

     def get_data(self):
         
         if self.buffer_counter < self.buffer_size:
             sample_ids = torch.randperm(self.buffer_counter)[: self.samples_from_buffer]
         else:
             sample_ids = torch.randperm(self.buffer_size)[: self.samples_from_buffer]
         
         return self.buffer_x[sample_ids], self.buffer_z[sample_ids], self.buffer_y[sample_ids]
     
        
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_train_y[self.current_task_id - self.num_old_task_window] 
         
         del self.task_test_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_test_y[self.current_task_id - self.num_old_task_window]
        
        

     




