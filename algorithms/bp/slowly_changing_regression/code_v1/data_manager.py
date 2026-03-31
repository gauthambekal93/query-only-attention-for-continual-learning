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
     
     def __init__(self, device, root, data_dir, flip_after, num_data_points, num_old_task_window, num_datapoints_per_timestep, train_size): 
         
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
         
     def create_scr_data(self):
        
         with open(self.data_path, 'rb+') as f:  #get the input and output features for training  inputs.shape torch.Size([10010000, 20]), outputs.shape torch.Size([10010000, 1])
             self.inputs, self.outputs, _ = pickle.load(f)  
             
                  
     def create_task_data(self):
             
             temp = self.inputs[self.current_task_id* self.flip_after :self.current_task_id* self.flip_after + self.flip_after]
             
             self.task_train_x[self.current_task_id] , self.task_test_x[self.current_task_id]  = temp[:self.train_size].to(self.device),  temp[self.train_size:].to(self.device) 
             
             temp = self.outputs[self.current_task_id* self.flip_after :self.current_task_id* self.flip_after + self.flip_after]
             
             self.task_train_y[self.current_task_id], self.task_test_y[self.current_task_id]  = temp[:self.train_size].to(self.device),  temp[self.train_size:].to(self.device) 


     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_train_y[self.current_task_id - self.num_old_task_window] 

         del self.task_test_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_test_y[self.current_task_id - self.num_old_task_window] 
         