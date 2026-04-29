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
     
     def __init__(self, device, root, data_dir, flip_after, num_data_points, num_old_task_window, num_datapoints_per_timestep, train_size, fifo_buffer_size, fifo_samples, data_input_dim): 
         
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
         
         
         self.fifo_x = torch.zeros(fifo_buffer_size , data_input_dim).to(self.device) 
         self.fifo_y = torch.zeros(fifo_buffer_size, 1).to(self.device).long()  
         self.fifo_buffer_size = fifo_buffer_size
         self.fifo_samples = fifo_samples
        
         self.fifo_counter = 0
        
        
     def create_scr_data(self):
        
         with open(self.data_path, 'rb+') as f:  #get the input and output features for training  inputs.shape torch.Size([10010000, 20]), outputs.shape torch.Size([10010000, 1])
             self.inputs, self.outputs, _ = pickle.load(f)  
             
                  
     def create_task_data(self):
             
             temp = self.inputs[self.current_task_id* self.flip_after :self.current_task_id* self.flip_after + self.flip_after]
             
             self.task_train_x[self.current_task_id] , self.task_test_x[self.current_task_id]  = temp[:self.train_size].to(self.device),  temp[self.train_size:].to(self.device) 
             
             temp = self.outputs[self.current_task_id* self.flip_after :self.current_task_id* self.flip_after + self.flip_after]
             
             self.task_train_y[self.current_task_id], self.task_test_y[self.current_task_id]  = temp[:self.train_size].to(self.device),  temp[self.train_size:].to(self.device) 


     def fill_fifo_buffer(self, x, y ):
            
            i = self.fifo_counter % len(self.fifo_x)
            
            self.fifo_x[ i : i + x.shape[0]], self.fifo_y[i: i + x.shape[0]] = x.clone(), y.clone()
            
            self.fifo_counter = self.fifo_counter + x.shape[0]
            
            
     def get_fifo_data(self):
                
            rand_ids = torch.randperm(  self.fifo_buffer_size)[:self.fifo_samples]
            
            support_x, support_y = self.fifo_x[rand_ids], self.fifo_y[rand_ids]
            
            return support_x, support_y 
        
        
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_train_y[self.current_task_id - self.num_old_task_window] 

         del self.task_test_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_test_y[self.current_task_id - self.num_old_task_window] 
         