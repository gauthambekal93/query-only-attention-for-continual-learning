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
     
     def __init__(self, root, data_dir, num_images_per_class, initial_num_classes, class_increase_per_task, total_classes, num_labels_previous_task, num_data_points_current_task , num_support_per_label, device):
         
         self.device = device
         self.initial_num_classes = initial_num_classes
         self.current_num_classes = initial_num_classes
         self.class_increase_per_task = class_increase_per_task
         self.total_classes = total_classes
         self.current_task_id = 0
         
         self.num_label_rows  = 5
         
         self.num_train_classes = initial_num_classes * ( self.num_label_rows + 1 )
         
         self.total_tasks = int( self.total_classes / self.class_increase_per_task )
         
         self.num_images_per_task = num_images_per_class * class_increase_per_task
         
         self.data_path = os.path.join( root, data_dir)
         
         self.pad = 4 
         
         #self.test_support_x = {}
         
         
         self.num_data_points_current_task =num_data_points_current_task
         
         self.num_support_per_label = num_support_per_label
         
         self.train_x, self.train_y,  self.test_x, self.test_y  = [] , [] , [], [] 
         
         self.task_train_x, self.task_train_y, self.task_val_x, self.task_val_y ,self.task_test_x, self.task_test_y = {}, {}, {}, {}, {}, {}
         
         self.buffer = torch.empty(self.total_classes, 5, 3, 32, 32).to(self.device) #{ i: torch.empty( 0 ) for i in range(self.total_classes)}
           
         self.counter = torch.zeros(self.total_classes, dtype = torch.int64)
         
         self.is_filled =  [0] * self.total_classes
         
         
         #self.num_pos_support_per_label, self.num_neg_support_per_label,  self.num_neg_support_labels = 1, 1, 40  # was ( 5, 5, 4), 2, 2, 10 
         
         #self.total_sample_supports = self.num_pos_support_per_label  + self.num_neg_support_per_label * self.num_neg_support_labels
         
         
         #self.support_y = torch.cat([ torch.tensor([j] * self.num_pos_support_per_label) if j == 0 else torch.tensor([j] * self.num_neg_support_per_label) 
         #                            for j in range(self.num_neg_support_labels + 1) ])
             
         #self.support_y = F.one_hot( self.support_y, num_classes = self.num_neg_support_labels + 1  ).to(self.device)  
         
         
         
         #self.supports_x = torch.empty(100, self.total_sample_supports, 3, 32, 32).to(self.device) 
         
         #self.supports_y = torch.empty(100, self.total_sample_supports,  self.num_neg_support_labels + 1 ).to(self.device) 
              
         
         self.test_support_y = torch.cat([ torch.tensor([j] *self.buffer.shape[1]) for j in range(self.total_classes) ])
             
         self.test_support_y = F.one_hot( self.test_support_y, num_classes = self.total_classes   ).to(self.device)  

         
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
        

     
     def fill_buffer(self, X, Y):
       
         for i, label in enumerate( Y ):
             label = label.item()
             
             self.buffer[label][ self.counter[label] ].copy_(X[i]) 
             
             self.counter[label] += 1
             
             if self.counter[label] == self.buffer.shape[1]:
                 
                 self.is_filled[label] = 1
                 
                 self.counter[label] = 0
                 

     def get_eval_support(self):
         
             support_x = self.buffer[:, :self.buffer.shape[1]]
             
             support_x = support_x.reshape(-1, support_x.shape[2], support_x.shape[3], support_x.shape[4])
             
             return support_x, self.test_support_y
         
            
     
     
     
     def create_unique_labels(self):
         
         self.unique_labels = torch.randperm(self.total_classes)[:5].to(self.device)
         
         self.mapping = {label.item(): i for i, label in enumerate(self.unique_labels)}
         
         
     def get_data(self):
         
         num_queries_per_label, num_support_per_label = 2, 1
         
         all_ids = torch.stack( [torch.randperm(self.buffer.shape[1])[: num_queries_per_label + num_support_per_label ] 
                                 for _ in range(len(self.unique_labels) ) ], dim = 0 )
         
         """Get Queries """
         query_x = self.buffer[self.unique_labels[:, None], all_ids[:, :num_queries_per_label]] 
            
         query_x = query_x.reshape(-1, query_x.shape[2], query_x.shape[3], query_x.shape[4])
         
         query_x = self.augment_batch(query_x)
         
         query_y =  self.unique_labels.repeat_interleave(num_queries_per_label)
         
         query_y = torch.tensor([self.mapping[l.item()] for l in query_y], device=self.device)
         
         rand_idx = torch.randperm(len(query_x))
         
         query_x, query_y = query_x[rand_idx], query_y[rand_idx]
         
         
         """Create Support """
         support_x = self.buffer[self.unique_labels[:, None], all_ids[:, num_queries_per_label:]] 
         
         support_x = support_x.reshape(-1, support_x.shape[2], support_x.shape[3], support_x.shape[4])
         
         support_y =  self.unique_labels.repeat_interleave(num_support_per_label)
         
         support_y = torch.tensor([self.mapping[l.item()] for l in support_y], device=self.device)
         
         support_y = F.one_hot( support_y, num_classes = len(self.unique_labels)  ).to(self.device)  
         
         return query_x, query_y, support_x, support_y
        
        
     def create_task_data(self):
    
             """Choose Random Labels for the task"""
             task_labels = torch.randperm(self.total_classes)[: self.initial_num_classes ].to(self.device)
             
             """Create train and validation """
             mask = torch.isin( self.train_y,  task_labels)
            
             rand_ids = torch.randperm(self.train_y [ mask ] .shape[0])
        
             train_val_ratio = 0.80
            
             train_rand_ids = rand_ids[: int( len(rand_ids) * train_val_ratio ) ]
            
             val_rand_ids= rand_ids [ int( len(rand_ids) * train_val_ratio ) : ]
            
             self.task_train_x[self.current_task_id] = self.augment_batch ( self.train_x[mask][train_rand_ids] )
             
             self.unique_labels = self.train_y [ mask ][train_rand_ids] .unique()
             
             self.task_train_y[self.current_task_id] = self.train_y [ mask ][train_rand_ids]
             
             
                
             """Create validation data """
             self.task_val_x[self.current_task_id] = self.train_x [mask][val_rand_ids]
             
             self.task_val_y[self.current_task_id] =  self.train_y [ mask ][val_rand_ids] 
        
             
             """Create test data """
             mask = torch.isin( self.test_y,  task_labels)
           
             self.task_test_x[self.current_task_id] = self.test_x [mask]
            
             self.task_test_y[self.current_task_id] = self.test_y [ mask ]
            
        

     
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - 20]
         
         del self.task_train_y[self.current_task_id - 20] 
         
         del self.task_val_x[self.current_task_id - 20]
         
         del self.task_val_y[self.current_task_id - 20]
         
         del self.task_test_x[self.current_task_id - 20]
         
         del self.task_test_y[self.current_task_id - 20]
        
        

     
     
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

     
     



