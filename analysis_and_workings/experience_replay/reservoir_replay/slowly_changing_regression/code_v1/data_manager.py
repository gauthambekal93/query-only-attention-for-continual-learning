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
import pickle

class DataManager:
     
     def __init__(self, device, root, data_dir, flip_after, num_data_points, num_old_task_window, num_datapoints_per_timestep, train_size, buffer_size, samples_from_buffer): 
         
         self.device = device
         self.data_path = os.path.join( root, data_dir)
        
         self.flip_after =  flip_after
         self.num_data_points = num_data_points
         self.num_old_task_window = num_old_task_window
         self.num_datapoints_per_timestep = num_datapoints_per_timestep
         self.current_task_id = 0
         self.num_tasks = int(self.num_data_points / self.flip_after)
         self.train_size = train_size
         self.task_train_x, self.task_train_y ,self.task_test_x, self.task_test_y = {}, {}, {}, {}
         
         self.buffer_x = torch.empty(buffer_size, 20).to(self.device)  
         self.buffer_y = torch.empty(buffer_size).to(self.device)
         
         self.step = 0
         self.buffer_counter = 0
         self.buffer_size = buffer_size
         self.samples_from_buffer = samples_from_buffer
         
         
     def create_scr_data(self):
        
         with open(self.data_path, 'rb+') as f:  #get the input and output features for training  inputs.shape torch.Size([10010000, 20]), outputs.shape torch.Size([10010000, 1])
             self.inputs, self.outputs, _ = pickle.load(f)  
             
                  
     def create_task_data(self):
             
             temp = self.inputs[self.current_task_id* self.flip_after :self.current_task_id* self.flip_after + self.flip_after]
             
             self.task_train_x[self.current_task_id] , self.task_test_x[self.current_task_id]  = temp[:self.train_size].to(self.device),  temp[self.train_size:].to(self.device) 
             
             temp = self.outputs[self.current_task_id* self.flip_after :self.current_task_id* self.flip_after + self.flip_after]
             
             self.task_train_y[self.current_task_id], self.task_test_y[self.current_task_id]  = temp[:self.train_size].to(self.device),  temp[self.train_size:].to(self.device) 
    
     
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
         
         return self.buffer_x[sample_ids], self.buffer_y[sample_ids].reshape(-1, 1)
     
        
     
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_train_y[self.current_task_id - self.num_old_task_window] 

         del self.task_test_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_test_y[self.current_task_id - self.num_old_task_window] 
         