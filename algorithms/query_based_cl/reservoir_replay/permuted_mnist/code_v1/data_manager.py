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


class DataManager:
     
     def __init__(self, device, root, data_dir, classes_per_task, num_old_task_window, buffer_size, num_datapoints_per_timestep, samples_from_buffer, num_tasks):    
         
         self.device = device
         self.data_path = os.path.join( root, data_dir)
         self.classes_per_task = classes_per_task
         self.num_old_task_window = num_old_task_window
         self.num_tasks = num_tasks
         self.current_task_id = 0
         
         self.task_train_x, self.task_train_y = {}, {}
         self.task_test_x, self.task_test_y = {}, {}
         
         self.buffer_x = torch.empty(buffer_size, 49).to(self.device)  
         self.buffer_y = torch.empty(buffer_size).to(self.device).long() 
         
         self.fifo_x, self.fifo_y = torch.empty(num_datapoints_per_timestep, 49).to(self.device) , torch.empty(num_datapoints_per_timestep).to(self.device)  
         
         self.step = 0
         self.buffer_counter = 0
         
         self.buffer_size = buffer_size
         
         self.num_datapoints_per_timestep = num_datapoints_per_timestep
         
         self.total_slot_ids = int( self.buffer_size / self.num_datapoints_per_timestep )
         
         self.samples_from_buffer = samples_from_buffer
      


     def create_permute_mnist_data(self):
                    
        with open( self.data_path , 'rb') as f:
            
            self.train_x, self.train_y, self.test_x, self.test_y = pickle.load(f)
            
            self.train_x = self.train_x.to(self.device)
            self.train_y = self.train_y.to(self.device).long()
            
            self.test_x = self.test_x.to(self.device)
            self.test_y = self.test_y.to(self.device).long()
     
            
     def create_task_data(self):
            
            
        pixel_permutation = torch.randperm(self.train_x.shape[1], device=self.device)
            
        data_permutation = torch.randperm(self.train_x.shape[0], device=self.device)
            
        self.task_train_x[self.current_task_id] = self.train_x[:, pixel_permutation][data_permutation]
            
        self.task_train_y[self.current_task_id] = self.train_y[data_permutation]
           
        self.task_test_x[self.current_task_id] = self.test_x[:, pixel_permutation]
            
        self.task_test_y[self.current_task_id] = self.test_y

         
     def fill_fifo_buffer(self, x, y ):
            self.fifo_x = x
            self.fifo_y = y
            
            
     def fill_reservoir_buffer(self, x, y):
         B = x.size(0)
         for i in range(B):
             self.step += 1
     
             if self.buffer_counter < self.buffer_size:
                 # fill phase
                 self.buffer_x[self.buffer_counter].copy_(x[i])
                 self.buffer_y[self.buffer_counter] = y[i]
                 self.buffer_counter += 1
             else:
                 # reservoir step
                 j = torch.randint(0, self.step, () ).item()  
                 if j < self.buffer_size:
                     self.buffer_x[j].copy_(x[i])
                     self.buffer_y[j] = y[i]
     

     def fill_buffer(self, x, y):
         self.fill_fifo_buffer( x, y )
         self.fill_reservoir_buffer(x, y)
         

     def get_data(self, buffer_type):
            
            if buffer_type =="fifo":
                X, Y = self.fifo_x, self.fifo_y
                k = 20

            if buffer_type =="reservoir":
                X, Y = self.buffer_x, self.buffer_y
                k = 20
            
            
            
            support_x, support_y = [], []
    
            unique_labels = torch.unique(Y)
            
            # samples per label
            for label in unique_labels:
                ids = (Y == label).nonzero(as_tuple=True)[0]
                
                # handle edge case (less than k samples)
                num_samples = min(k, ids.size(0))
                
                rand_ids = ids[torch.randperm(ids.size(0))[:num_samples]]
                
                support_x.append(X[rand_ids])
                support_y.append(Y[rand_ids])
                
            support_x = torch.cat(support_x, dim=0)
            support_y = torch.cat(support_y, dim=0)
            support_y = F.one_hot( support_y, num_classes = len(unique_labels)  ).to(self.device)  
            
            return support_x, support_y
          
         
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_train_y[self.current_task_id - self.num_old_task_window] 

         del self.task_test_x[self.current_task_id - self.num_old_task_window]
        
         del self.task_test_y[self.current_task_id - self.num_old_task_window] 
        
  
   
     



