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
import pickle
from augmentations import get_task_params, augment_batch

class DataManager:
     
     def __init__(self, device, root, data_dir, classes_per_task, total_classes, num_old_task_window, num_datapoints_per_timestep, num_tasks, per_task_buffer_size, fifo_buffer_size, fifo_samples, balanced_task_samples): 
         
          
         self.device = device
         self.data_path = os.path.join( root, data_dir)
         self.classes_per_task = classes_per_task
         self.total_classes = total_classes
         self.num_old_task_window = num_old_task_window
         self.num_tasks = num_tasks
         self.current_task_id = 0
         
         self.train_x, self.train_y,  self.test_x, self.test_y  = [] , [] , [], [] 
         
         self.task_train_x, self.task_train_y ,self.task_test_x, self.task_test_y = {}, {}, {}, {}
         
         
         self.balanced_task_buffer_x = { i: torch.empty( per_task_buffer_size , 3, 32, 32).to(self.device) for i in range(self.num_old_task_window )}  
         self.balanced_task_buffer_y = { i: torch.empty( per_task_buffer_size).to(self.device).long() for i in range(self.num_old_task_window ) }
         self.balanced_task_samples = balanced_task_samples
                                                 
         self.fifo_x = torch.zeros(2, fifo_buffer_size , 3, 32, 32).to(self.device) 
         self.fifo_y = torch.zeros(2, fifo_buffer_size).to(self.device).long()  
         self.fifo_samples = fifo_samples
         
         self.buffer_key_counter = 0
         self.buffer_counter = 0
         self.fifo_counter_0, self.fifo_counter_1 = 0, 0
        
         self.num_datapoints_per_timestep = num_datapoints_per_timestep
         
        

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
             
            
             data_permutation = torch.randperm(train_x.shape[0], device=self.device)
             
             params = get_task_params(self.current_task_id)
             
             train_x = augment_batch(train_x, params)

             
             train_x = train_x[data_permutation]
             
             train_y = train_y[data_permutation]
             
             self.task_train_x[self.current_task_id] = train_x
            
             self.task_train_y[self.current_task_id] = self.relable_data(train_y, task_labels)
             
             
             mask = torch.isin( self.test_y,  task_labels)
             
             test_x , test_y =  self.test_x[mask], self.test_y[mask]

             
             test_x = augment_batch(test_x, params)
             
             self.task_test_x[self.current_task_id] = test_x
            
             self.task_test_y[self.current_task_id] = self.relable_data(test_y, task_labels)    
     
        
     def fill_fifo_buffer_1(self, x, y ):
            
            i = self.fifo_counter_0 % len(self.fifo_x[1])
            
            self.fifo_x[1, i : i + x.shape[0]], self.fifo_y[1, i: i + x.shape[0]] = x.clone(), y.clone()
            
            self.fifo_counter_0 = self.fifo_counter_0 + x.shape[0]
           
            
     def fill_fifo_buffer_0(self, x, y ):
           
           j = self.fifo_counter_1 % len(self.fifo_x[0])
         
           self.fifo_x[0, j : j + x.shape[0]], self.fifo_y[0, j: j + x.shape[0]]  = x.clone(), y.clone()
           
           self.fifo_counter_1 = self.fifo_counter_1 + x.shape[0]
           
           
     def fill_balaced_task_buffer(self, x, y, buffer_key):
         
         i = self.buffer_counter % self.balanced_task_buffer_x[0].shape[0]
         
         self.balanced_task_buffer_x[buffer_key][ i : i + x.shape[0] ] = x.clone()
         self.balanced_task_buffer_y[buffer_key][ i : i + x.shape[0] ] = y.clone()
         
         self.buffer_counter = self.buffer_counter + x.shape[0]


     def get_fifo_data(self, fifo_id = 1):
            
            #if self.current_task_id==19 and i>=1230:
            #    print(i)
            #    print("stop")
                
            X, Y = self.fifo_x[fifo_id], self.fifo_y[fifo_id]
            
            support_x, support_y = [], []
    
            unique_labels = torch.arange(self.classes_per_task) #torch.unique(Y)
            
            # samples per label
            for label in unique_labels:
                matched_ids = (Y == label).nonzero(as_tuple=True)[0]
                
                # handle edge case (less than k samples)
                num_samples = min(self.fifo_samples, matched_ids.size(0))
                
                rand_ids = matched_ids[torch.randperm(matched_ids.size(0))[:num_samples]]
                
                #selected_ids = matched_ids[-num_samples:]
                
                support_x.append(X[rand_ids])
                support_y.append(Y[rand_ids])
                
            support_x = torch.cat(support_x, dim=0)
            support_y = torch.cat(support_y, dim=0)
            support_y = F.one_hot( support_y, num_classes = len(unique_labels)  ).to(self.device)  
            
            #if len(support_y)<10:
            #    print("stop")
            #    print("stop")
            
            return support_x, support_y
        
            
     def get_balaced_task_data(self, X, Y, unique_labels):
         
         support_x, support_y = [], []
             # samples per label
         for label in unique_labels:
            ids = (Y == label).nonzero(as_tuple=True)[0]
            
            # handle edge case (less than k samples)
            num_samples = min(self.balanced_task_samples, ids.size(0))
            
            rand_ids = ids[torch.randperm(ids.size(0))[:num_samples]]
            
            support_x.append(X[rand_ids])
            support_y.append(Y[rand_ids])
                 
         support_x = torch.cat(support_x, dim=0)
         support_y = torch.cat(support_y, dim=0)
         support_y = F.one_hot( support_y, num_classes = self.classes_per_task  ).to(self.device)      
             
         return support_x, support_y    
             
             
        
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_train_y[self.current_task_id - self.num_old_task_window] 

         del self.task_test_x[self.current_task_id - self.num_old_task_window]
        
         del self.task_test_y[self.current_task_id - self.num_old_task_window] 
        
  
   
     



