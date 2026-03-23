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
         self.buffer_y = torch.empty(buffer_size).to(self.device).long() 
         
         self.step = 0
         self.buffer_counter = 0
         
         self.buffer_size = buffer_size
         self.samples_from_buffer = samples_from_buffer
         

         
     def create_cifar_data(self):
        
        """The numbers are mean and std across 3 channels of the image.
            I have confirmed these mean and std values are correct, 
            by initailly downloading and manually inspecting meand and std"""
     
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))   
        ])
        
        self.train_set = datasets.CIFAR100(
            root=self.data_path ,
            train=True,
            download=False,  
            transform=transform
        )
     
        self.test_set = datasets.CIFAR100(
            root=self.data_path ,
            train=False,
            download=False,  
            transform=transform
        )
       

        for img, label in self.train_set:
            self.train_x.append(img)
            self.train_y.append(label)
            
        self.train_x = torch.stack (self.train_x).to(self.device)
        
        self.train_y = torch.tensor(self.train_y).to(self.device)
        
        
        for img, label in self.test_set:
            self.test_x.append(img)
            self.test_y.append(label)
            
        self.test_x = torch.stack (self.test_x).to(self.device)
        
        self.test_y = torch.tensor(self.test_y).to(self.device)
        

     
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

        

     def fill_buffer(self, x, y):
        
         B = x.size(0)
         for i in range(B):
             self.step += 1
     
             if self.buffer_counter < self.buffer_size:
                 # fill phase
                 self.buffer_x[self.buffer_counter].copy_(x[i].clone())
                 self.buffer_y[self.buffer_counter] = y[i].clone()
                 self.buffer_counter += 1
             else:
                 # reservoir step
                 j = torch.randint(0, self.step, () ).item()  
                 if j < self.buffer_size:
                     self.buffer_x[j].copy_(x[i].clone())
                     self.buffer_y[j] = y[i].clone()                     
                     

     def get_data(self):
         
         if self.buffer_counter < self.buffer_size:
             sample_ids = torch.randperm(self.buffer_counter)[: self.samples_from_buffer]
         else:
             sample_ids = torch.randperm(self.buffer_size)[: self.samples_from_buffer]
         
         return self.buffer_x[sample_ids], self.buffer_y[sample_ids]
     
        
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_train_y[self.current_task_id - self.num_old_task_window] 
         
         del self.task_test_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_test_y[self.current_task_id - self.num_old_task_window]
        
        

     




