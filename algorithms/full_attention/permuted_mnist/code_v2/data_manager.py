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
     
     def __init__(self, device, root, data_dir, classes_per_task, num_old_task_window, buffer_size, num_datapoints_per_timestep, samples_per_label, num_tasks):    
         
         self.device = device
         self.data_path = os.path.join( root, data_dir)
         self.classes_per_task = classes_per_task
         self.num_old_task_window = num_old_task_window
         self.num_tasks = num_tasks
         self.current_task_id = 0
         
         self.task_train_x, self.task_train_y = {}, {}
         self.task_test_x, self.task_test_y = {}, {}
         
         #self.buffer_x = torch.empty(buffer_size, 49).to(self.device)  
         #self.buffer_y = torch.empty(buffer_size).to(self.device).long() 
         
         #self.buffer_x = { i: torch.empty( num_datapoints_per_timestep, 49).to(self.device) for i in range(self.num_old_task_window )}  
         #self.buffer_y = { i: torch.empty( num_datapoints_per_timestep).to(self.device).long() for i in range(self.num_old_task_window ) }
                                                          
         self.fifo_x, self.fifo_y = torch.zeros(buffer_size , 49).to(self.device) , torch.zeros(buffer_size).to(self.device).long()  
         
         #self.new_task = True
         
         #self.buffer_counter = 0
         
         self.buffer_size = buffer_size
         
         self.num_datapoints_per_timestep = num_datapoints_per_timestep
         
         #self.total_slot_ids = int( self.buffer_size / self.num_datapoints_per_timestep )
         
         self.samples_per_label = samples_per_label
         self.fifo_counter= 0


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
            
            i = self.fifo_counter % len(self.fifo_x)
            
            self.fifo_x[i : i + x.shape[0]], self.fifo_y[i: i + x.shape[0]] = x.clone(), y.clone()
            
            self.fifo_counter = self.fifo_counter + x.shape[0]
           
    

     def get_fifo_data(self):
            
            X, Y = self.fifo_x.clone(), self.fifo_y.clone()

            support_x, support_y = [], []
    
            unique_labels = torch.unique(Y)
            
            # samples per label
            for label in unique_labels:
                ids = (Y == label).nonzero(as_tuple=True)[0]
                
                # handle edge case (less than k samples)
                num_samples = min( self.samples_per_label , ids.size(0))
                
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
        
  
   
     



