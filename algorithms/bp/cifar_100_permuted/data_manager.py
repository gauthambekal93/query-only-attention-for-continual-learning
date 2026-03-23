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


class DataManager:
     
     #def __init__(self, root, data_dir, num_images_per_class, initial_num_classes, class_increase_per_task, total_classes, num_labels_previous_task, num_data_points_current_task , num_support_per_label, device):
     def __init__(self, device, root, data_dir, classes_per_task, total_classes, num_old_task_window, num_datapoints_per_timestep, num_tasks):    
       
         self.device = device
         self.classes_per_task = classes_per_task
         #self.current_num_classes = initial_num_classes
         #self.class_increase_per_task = class_increase_per_task
         self.total_classes = total_classes
         self.current_task_id = 0
         self.num_old_task_window = num_old_task_window
         #self.num_label_rows  = 5
         
         #self.num_train_classes = initial_num_classes * ( self.num_label_rows + 1 )
         
         self.num_tasks = num_tasks #int( self.total_classes / self.class_increase_per_task )
         
         #self.num_images_per_task = num_images_per_class * class_increase_per_task
         
         self.data_path = os.path.join( root, data_dir)
         
         self.pad = 4 
        
         #self.num_data_points_current_task =num_data_points_current_task
         
         #self.num_support_per_label = num_support_per_label
         
         self.train_x, self.train_y,  self.test_x, self.test_y  = [] , [] , [], [] 
         
         self.task_train_x, self.task_train_y ,self.task_test_x, self.task_test_y = {}, {}, {}, {}
         
         #self.buffer = torch.empty(self.total_classes, 20, 3, 32, 32).to(self.device) 
           
         #self.counter = torch.zeros(self.total_classes, dtype = torch.int64)
         
         #self.is_filled =  [0] * self.total_classes
         

         
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
        

     '''
     def fill_buffer(self, X, Y):
       
         for i, label in enumerate( Y ):
             label = label.item()
             
             self.buffer[label][ self.counter[label] ].copy_(X[i]) 
             
             self.counter[label] += 1
             
             if self.counter[label] == 20:
                 
                 self.is_filled[label] = 1
                 
                 self.counter[label] = 0
                 

     
     def create_unique_labels(self):
         
         num_unique_labels = 10
         
         self.unique_labels = torch.randperm(self.total_classes)[:num_unique_labels].to(self.device)
         
    
              
     def get_data(self):

         num_datapoints_per_label = 1
         
         all_ids = torch.stack( [torch.randperm(self.buffer.shape[1])[:num_datapoints_per_label ] for _ in range(len(self.unique_labels) ) ], dim = 0 )
         
         replay_x = self.buffer[self.unique_labels[:, None], all_ids[:, :num_datapoints_per_label]] 
         
         replay_x = replay_x.reshape(  -1, replay_x.shape[2], replay_x.shape[3], replay_x.shape[4])
         
         #replay_x = self.augment_batch(replay_x)
         
         replay_y =  self.unique_labels.repeat_interleave(num_datapoints_per_label)
            
         rand_ids = torch.randperm(len(replay_x))
         
         replay_x, replay_y = replay_x[rand_ids], replay_y[rand_ids]
         
         return replay_x, replay_y
     
     '''
     
     def relable_data(self, Y, task_labels):
         
            Y_new = torch.empty_like(Y)
        
            for new_label, old_label in enumerate(task_labels):
                Y_new[Y == old_label] = new_label
        
            return Y_new
                 
                  
     def create_task_data(self):
            
             task_labels = torch.randperm(self.total_classes)[: self.classes_per_task ].to(self.device)
             
             
             mask = torch.isin( self.train_y,  task_labels)
             
             train_x, train_y = self.train_x[mask], self.train_y[mask]
             
             pixel_permutation = torch.randperm(train_x.shape[2], device=self.device)
            
             data_permutation = torch.randperm(train_x.shape[0], device=self.device)
            
             train_x = train_x[:, :, pixel_permutation, :]
             
             train_x = train_x[:, :, :, pixel_permutation]
             
             train_x = train_x[data_permutation]
             
             train_y = train_y[data_permutation]
             
             self.task_train_x[self.current_task_id] = train_x
            
             self.task_train_y[self.current_task_id] = self.relable_data(train_y, task_labels)
             
             
             mask = torch.isin( self.test_y,  task_labels)
             
             test_x , test_y =  self.test_x[mask], self.test_y[mask]
             
             test_x = test_x[:, :, pixel_permutation, :]
             
             test_x = test_x[:, :, :, pixel_permutation]
             
             self.task_test_x[self.current_task_id] = test_x
            
             self.task_test_y[self.current_task_id] = self.relable_data(test_y, task_labels) 

        

     
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_train_y[self.current_task_id - self.num_old_task_window] 
         
         #del self.task_val_x[self.current_task_id - self.num_old_task_window]
         
         #del self.task_val_y[self.current_task_id - self.num_old_task_window]
         
         del self.task_test_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_test_y[self.current_task_id - self.num_old_task_window]
        
        

     
     
     def augment_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,32,32] normalized tensors on GPU
        returns: augmented x, same shape
        """
        # RandomHorizontalFlip (p=0.5) per-image
        B = x.size(0)
        flip_mask = torch.rand(B, device=x.device) < 0.5
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])  # flip width
    
        # RandomCrop(size=32, padding=4, reflect)
        # reflect pad: [B,3,32+8,32+8] = [B,3,40,40]
        x = torch.nn.functional.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
    
        # choose crop offsets per image
        max_off = 2 * self.pad  # 8
        off_y = torch.randint(0, max_off + 1, (B,), device=x.device)
        off_x = torch.randint(0, max_off + 1, (B,), device=x.device)
    
        # crop each image back to 32x32
        crops = []
        for i in range(B):
            y = off_y[i].item()
            xx = off_x[i].item()
            crops.append(x[i:i+1, :, y:y+32, xx:xx+32])
        x = torch.cat(crops, dim=0)
    
        # RandomRotator(degrees=(0,15)) per-image
        # NOTE: rotation on GPU uses TF.rotate which expects CPU sometimes depending on backend.
        # Easiest: do it on CPU in your dataloader. If you insist on pure tensor-GPU, skip rotation.
        return x

     
     



