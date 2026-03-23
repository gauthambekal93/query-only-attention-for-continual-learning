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
         
         self.task_train_x, self.task_train_y, self.task_train_y_one_hot = {}, {}, {}
         self.task_test_x, self.task_test_y, self.task_test_y_one_hot = {}, {}, {}
         
         self.buffer_x = torch.empty(buffer_size, 49).to(self.device)  
         self.buffer_y = torch.empty(buffer_size).to(self.device).long() 
         #self.buffer_y_one_hot =  torch.empty(buffer_size, 10).to(self.device)
         
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
            #self.train_y_one_hot = F.one_hot(self.train_y, num_classes=self.classes_per_task).float()
    
            
            self.test_x = self.test_x.to(self.device)
            self.test_y = self.test_y.to(self.device).long()
            #self.test_y_one_hot = F.one_hot(self.test_y, num_classes=self.classes_per_task).float()  
     
            
     def create_task_data(self):
         
         pixel_permutation = torch.randperm(self.train_x.shape[1], device=self.device)
         
         data_permutation = torch.randperm(self.train_x.shape[0], device=self.device)
         
         if self.current_task_id ==0:
                               
             self.task_train_x[self.current_task_id] = self.train_x[:, pixel_permutation][data_permutation]
             
             self.task_train_y[self.current_task_id] = self.train_y[data_permutation]
            
             #self.task_train_y_one_hot[self.current_task_id] = self.train_y_one_hot[data_permutation]
            
            
             self.task_test_x[self.current_task_id] = self.test_x[:, pixel_permutation]
             
             self.task_test_y[self.current_task_id] = self.test_y
            
             #self.task_test_y_one_hot[self.current_task_id] = self.test_y_one_hot
         
         
         else:
             
             self.task_train_x[self.current_task_id] = self.task_train_x[self.current_task_id - 1][:, pixel_permutation][data_permutation]
             
             self.task_train_y[self.current_task_id] = self.task_train_y[self.current_task_id - 1][data_permutation]
            
             #self.task_train_y_one_hot[self.current_task_id] = self.task_train_y_one_hot[self.current_task_id - 1][data_permutation]
            
            
             self.task_test_x[self.current_task_id] = self.task_test_x[self.current_task_id - 1][:, pixel_permutation]
             
             self.task_test_y[self.current_task_id] = self.task_test_y[self.current_task_id -1]
            
             #self.task_test_y_one_hot[self.current_task_id] = self.task_test_y_one_hot[self.current_task_id - 1]
             
         
         
         
     def fill_buffer(self, x, y):
       
         current_slot_id = int( self.buffer_counter % self.total_slot_ids )
         
         start = current_slot_id *  self.num_datapoints_per_timestep
         
         end = start + self.num_datapoints_per_timestep
         
         self.buffer_x[start: end] = x
         self.buffer_y[start: end] = y
         #self.buffer_y_one_hot[start: end] = y_one_hot
         
         self.buffer_counter = self.buffer_counter  + 1


     def get_data(self):
         
         sample_ids = torch.randperm(self.buffer_size)[: self.samples_from_buffer]
         
         return self.buffer_x[sample_ids], self.buffer_y[sample_ids], self.buffer_y_one_hot[sample_ids]
         
         
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_train_y[self.current_task_id - self.num_old_task_window] 
         
         #del self.task_train_y_one_hot[self.current_task_id - self.num_old_task_window] 


         del self.task_test_x[self.current_task_id - self.num_old_task_window]
        
         del self.task_test_y[self.current_task_id - self.num_old_task_window] 
        
         #del self.task_test_y_one_hot[self.current_task_id - self.num_old_task_window] 
  
   
     



