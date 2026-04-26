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
from augmentations import get_task_params, augment_batch
from torch.utils.data import DataLoader
import random
from torch.utils.data import DataLoader, Subset


class DataManager:
     
     def __init__(self, device, root, data_dir, classes_per_task, total_classes, num_old_task_window, num_datapoints_per_timestep, num_tasks, buffer_size, samples_from_buffer):    
       
         self.device = device
         self.classes_per_task = classes_per_task
    
         self.total_classes = total_classes
         self.current_task_id = 0
         self.num_old_task_window = num_old_task_window
         
         self.num_tasks = num_tasks
         
         self.data_path = os.path.join( root, data_dir)
         
         self.pad = 4 
        
         self.train_x, self.train_y,  self.test_x, self.test_y  = [] , [] , [], [] 
         
         self.task_train_x, self.task_train_y ,self.task_test_x, self.task_test_y = {}, {}, {}, {}
         
         self.buffer_x = torch.empty(buffer_size, 3, 128, 128).to(self.device) 
         self.buffer_z = torch.empty(buffer_size, classes_per_task).to(self.device)
         self.buffer_y = torch.empty(buffer_size).to(self.device).long() 
         
         self.step = 0
         self.buffer_counter = 0
         
         self.buffer_size = buffer_size
         self.samples_from_buffer = samples_from_buffer
         
         self.batch_size = 256
         self.img_size = 128
         self.num_workers = 1
  
         self._build_dataset()
         self._build_class_index()

         
     def _build_dataset(self):

        normalize = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            normalize,
        ])

        self.dataset = datasets.ImageFolder(
            root=self.data_path,
            transform=self.transform
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.num_classes = len(self.dataset.classes)

        print(f"Loaded ImageNet with {self.num_classes} classes")

         

     def _build_class_index(self):

        self.class_to_indices = {}

        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        print("Built class index")
        
     
     def sample_task(self ):

        samples_per_class = 500
        chosen_classes = random.sample(range(self.num_classes), self.classes_per_task)
   
        # collect indices
        all_indices = []
        class_counts = {}   # track counts per class

        for c in chosen_classes:
            indices = self.class_to_indices[c]
    
            if len(indices) > samples_per_class:
                chosen_idx = random.sample(indices, samples_per_class)
            else:
                chosen_idx = indices
    
            all_indices.extend(chosen_idx)
            class_counts[c] = len(chosen_idx)
    
        subset = Subset(self.dataset, all_indices)
       
        loader = DataLoader(
            subset,
            batch_size=512,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True
        )
        #print(time.time() - start)
        
        total_size = len(all_indices)
        #start = time.time() 
        # preallocate
        task_imgs = torch.empty((total_size, 3, self.img_size, self.img_size))
        task_labels = torch.empty((total_size,), dtype=torch.long)
    
        ptr = 0
        
        # load once
        for img, label in loader:
            B = img.size(0)
            task_imgs[ptr:ptr+B] = img
            task_labels[ptr:ptr+B] = label
            ptr += B
        #print(time.time() - start)
        # split
        train_x, train_y, test_x, test_y = [], [], [], []
        #start = time.time() 
        for c in chosen_classes:
            mask = (task_labels == c)
    
            class_imgs = task_imgs[mask]
            class_labels = task_labels[mask]
    
            n = class_imgs.size(0)
            n_train = int(0.8 * n)
    
            train_x.append(class_imgs[:n_train])
            train_y.append(class_labels[:n_train])
    
            test_x.append(class_imgs[n_train:])
            test_y.append(class_labels[n_train:])
    
        train_x = torch.cat(train_x).to(self.device)
        train_y = torch.cat(train_y).to(self.device)
    
        test_x = torch.cat(test_x).to(self.device)
        test_y = torch.cat(test_y).to(self.device)
       
        return train_x, train_y, test_x, test_y, chosen_classes
        
        
        

     
     def relable_data(self, Y, task_labels):
         
            Y_new = torch.empty_like(Y)
        
            for new_label, old_label in enumerate(task_labels):
                Y_new[Y == old_label] = new_label
        
            return Y_new
                 
                  
     def create_task_data(self):
            
             train_x, train_y, test_x, test_y, task_labels = self.sample_task()
             
             data_permutation = torch.randperm(train_x.shape[0], device=self.device)
             
             params = get_task_params(self.current_task_id)
             
             train_x = augment_batch(train_x, params)

             
             train_x = train_x[data_permutation]
             
             train_y = train_y[data_permutation]
             
             self.task_train_x[self.current_task_id] = train_x
            
             self.task_train_y[self.current_task_id] = self.relable_data(train_y, task_labels)
             
             
             test_x = augment_batch(test_x, params)
             
             self.task_test_x[self.current_task_id] = test_x
            
             self.task_test_y[self.current_task_id] = self.relable_data(test_y, task_labels)     

        

     def fill_buffer(self, x, z, y):
         
         z = z.detach()
         B = x.size(0)
         for i in range(B):
             self.step += 1
     
             if self.buffer_counter < self.buffer_size:
                 # fill phase
                 self.buffer_x[self.buffer_counter].copy_(x[i].clone())
                 self.buffer_y[self.buffer_counter] = y[i].clone()
                 self.buffer_z [self.buffer_counter].copy_(z[i].clone())
                 self.buffer_counter += 1
             else:
                 # reservoir step
                 j = torch.randint(0, self.step, () ).item()  
                 if j < self.buffer_size:
                     self.buffer_x[j].copy_(x[i].clone())
                     self.buffer_z[j].copy_(z[i].clone())
                     self.buffer_y[j] = y[i].clone()                     
                     

     def get_data(self):
         
         if self.buffer_counter < self.buffer_size:
             sample_ids = torch.randperm(self.buffer_counter)[: self.samples_from_buffer]
         else:
             sample_ids = torch.randperm(self.buffer_size)[: self.samples_from_buffer]
         
         return self.buffer_x[sample_ids], self.buffer_z[sample_ids], self.buffer_y[sample_ids]
     
        
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_train_y[self.current_task_id - self.num_old_task_window] 
         
         del self.task_test_x[self.current_task_id - self.num_old_task_window]
         
         del self.task_test_y[self.current_task_id - self.num_old_task_window]
        
        

     




