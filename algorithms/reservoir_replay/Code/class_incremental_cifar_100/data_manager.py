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
     
     def __init__(self, root, data_dir, num_images_per_class, initial_num_classes, total_classes , mixing_ratio, incoming_batch_size, task_lag, device):
         
         self.device = device
         self.initial_num_classes = initial_num_classes
         self.current_num_classes = initial_num_classes
         self.total_classes = total_classes
         self.current_task_id = 0
         
         self.num_label_rows  = 5
         
         self.num_train_classes = initial_num_classes * ( self.num_label_rows + 1 )
        
         self.data_path = os.path.join( root, data_dir)
         
         self.pad = 4 
         
         self.train_x, self.train_y,  self.test_x, self.test_y  = [] , [] , [], [] 
         
         self.task_train_x, self.task_train_y, self.task_val_x, self.task_val_y ,self.task_test_x, self.task_test_y = {}, {}, {}, {}, {}, {}
         
         # Build lookup once
         self.classes = torch.full((self.total_classes,), -1, device=self.device)  # CIFAR-100
        
         self.task_lag = task_lag
         
         self.buffer_size = 2000
         
         self.buffer_x = torch.empty( (self.buffer_size, 3, 32, 32 ) ).to(self.device)
         self.buffer_y = torch.empty( (self.buffer_size )   , dtype=torch.long ).to( self.device)
         self.step = 0
         self.buffer_count = 0
         
         self.mixing_ratio = mixing_ratio
         self.incoming_batch_size = incoming_batch_size
         
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
         
         if self.step < self.buffer_size - self.incoming_batch_size:
             
             self.buffer_x[self.step : self.step + self.incoming_batch_size] = X  
             
             self.buffer_y[self.step : self.step + self.incoming_batch_size] = Y
            
         else:
             try:
                 j = torch.randperm(self.step)[:1]
                 
                 if ( j  < self.buffer_size - self.incoming_batch_size) and ( j>=0 ) :
                     
                     #idx = self.step % ( self.buffer_x.shape[0]  + self.incoming_batch_size )   
                     
                     self.buffer_x[j : j + self.incoming_batch_size] = X
                     self.buffer_y[j : j + self.incoming_batch_size] = Y
             except:
                 print("stop")
                 print("stop")
                 
         self.step += self.incoming_batch_size
     
     '''
     
     def fill_buffer(self, X: torch.Tensor, Y: torch.Tensor):

        B = X.size(0)
        for i in range(B):
            self.step += 1
    
            if self.buffer_count < self.buffer_size:
                # fill phase
                self.buffer_x[self.buffer_count].copy_(X[i])
                self.buffer_y[self.buffer_count] = Y[i]
                self.buffer_count += 1
            else:
                # reservoir step
                j = torch.randint(0, self.step, () ).item()  
                if j < self.buffer_size:
                    self.buffer_x[j].copy_(X[i])
                    self.buffer_y[j] = Y[i]
    


     def sample_buffer(self):
         
          rand_idx = torch.randperm(self.buffer_size)[: int(self.incoming_batch_size * self.mixing_ratio) ]
          
          X, Y = self.buffer_x[rand_idx] , self.buffer_y[rand_idx]
          
          return X, Y
      
        
     def create_task_data(self):
    
             """Choose Random Labels for the task"""
             task_labels = torch.randperm(self.total_classes)[: self.initial_num_classes ].to(self.device)
             
             """Create train and validation """
             mask = torch.isin( self.train_y,  task_labels)
            
             rand_ids = torch.randperm(self.train_y [ mask ] .shape[0])
        
             train_val_ratio = 0.80
            
             train_rand_ids = rand_ids[: int( len(rand_ids) * train_val_ratio ) ]
            
             val_rand_ids= rand_ids [ int( len(rand_ids) * train_val_ratio ) : ]
            
             self.task_train_x[self.current_task_id] = self.train_x[mask][train_rand_ids]
              
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

     
     



