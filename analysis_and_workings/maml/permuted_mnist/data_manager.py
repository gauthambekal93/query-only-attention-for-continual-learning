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
     
     def __init__(self, device, root, data_dir, classes_per_task, num_old_task_window, num_datapoints_per_timestep, supports_per_task, queries_per_task, buffer_size_per_task, num_tasks_in_buffer, num_tasks):    
         
         self.device = device
         self.data_path = os.path.join( root, data_dir)
         self.classes_per_task = classes_per_task
         self.num_old_task_window = num_old_task_window
         self.num_tasks = num_tasks
         self.current_task_id = 0
         
         self.task_train_x, self.task_train_y = {}, {}
         self.task_test_x, self.task_test_y = {}, {}
         
         self.num_tasks_in_buffer = num_tasks_in_buffer
         self.buffer_size_per_task = buffer_size_per_task
         
         self.buffer_x = { i: torch.empty( buffer_size_per_task, 49).to(self.device) for i in range(num_tasks_in_buffer )}  
         self.buffer_y = { i: torch.empty( buffer_size_per_task).to(self.device).long() for i in range(num_tasks_in_buffer ) }
         
         #self.buffer_x = { }  
         #self.buffer_y = { }
         
         self.num_datapoints_per_timestep = num_datapoints_per_timestep
         
         self.supports_per_task = supports_per_task
         
         self.queries_per_task = queries_per_task
         
         self.fifo_counter = 0
         self.buffer_key = 0
         
     def create_permute_mnist_data(self):
                    
        with open( self.data_path , 'rb') as f:
            
            self.train_x, self.train_y, _, _ = pickle.load(f)
            
            self.train_x = self.train_x.to(self.device)
            self.train_y = self.train_y.to(self.device).long()
            

     
            
     def create_task_data(self, task_id):
            
        pixel_permutation = torch.randperm(self.train_x.shape[1], device=self.device)
            
        data_permutation = torch.randperm(self.train_x.shape[0], device=self.device)
            
        self.task_train_x[task_id] = self.train_x[:, pixel_permutation][data_permutation]
            
        self.task_train_y[task_id] = self.train_y[data_permutation]
           
  
   
     
     def fill_buffer(self, x, y ):
         
        self.buffer_x = x[ : self.buffer_size_per_task].clone()
        
        self.buffer_y = y[ : self.buffer_size_per_task].clone()
        

             
             
     def get_buffer_data(self, selected_task_ids):
         
         X, Y = self.buffer_x.clone(), self.buffer_y.clone()
         
         supports_x = X[:self.supports_per_task]
         
         supports_y = Y[:self.supports_per_task]
         
         queries_x  =  X[-self.queries_per_task:]
         
         queries_y  =  Y[-self.queries_per_task:]
         
         return supports_x , supports_y, queries_x, queries_y

            
             
        
     def delete_data(self, task_id):
         
         del self.task_train_x[task_id], self.task_train_y[task_id], self.task_train_x[task_id -1 ], self.task_train_y[task_id -1]
        
  
   
     



