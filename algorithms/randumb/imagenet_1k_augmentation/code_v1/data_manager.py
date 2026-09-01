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
from augmentations import get_task_params, augment_batch
from torch.utils.data import DataLoader
import random
from torch.utils.data import DataLoader, Subset

class DataManager:
     
     def __init__(self, device, root, data_dir, classes_per_task, total_classes, num_old_task_window, num_datapoints_per_timestep, num_tasks, fifo_buffer_size, fifo_samples_per_label): 
         
          
         self.device = device
         self.data_path = os.path.join( root, data_dir)
         self.classes_per_task = classes_per_task
         self.total_classes = total_classes
         self.num_old_task_window = num_old_task_window
         self.num_tasks = num_tasks
         self.current_task_id = 0
         
         self.train_x, self.train_y,  self.test_x, self.test_y  = [] , [] , [], [] 
         
         self.task_train_x, self.task_train_y ,self.task_test_x, self.task_test_y = {}, {}, {}, {}
         
         
         #self.balanced_task_buffer_x = { i: torch.empty( per_task_buffer_size , 3, 128, 128).to(self.device) for i in range(self.num_old_task_window )}  
         #self.balanced_task_buffer_y = { i: torch.empty( per_task_buffer_size).to(self.device).long() for i in range(self.num_old_task_window ) }
         #self.balanced_task_samples = balanced_task_samples
                                                 
         self.fifo_x = torch.zeros(fifo_buffer_size , 3, 128, 128).to(self.device) 
         self.fifo_y = torch.zeros(fifo_buffer_size).to(self.device).long()  
         self.fifo_samples_per_label = fifo_samples_per_label
         
         self.fifo_counter= 0
        
         self.num_datapoints_per_timestep = num_datapoints_per_timestep
         
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

        import time
        samples_per_class = 500
        chosen_classes = random.sample(range(self.num_classes), self.classes_per_task)
        start = time.time() 
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
        print(time.time() - start)
       
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

        
        
     def fill_fifo_buffer(self, x, y ):
            
            i = self.fifo_counter % len(self.fifo_x)
            
            self.fifo_x[ i : i + x.shape[0]], self.fifo_y[ i: i + x.shape[0]] = x.clone(), y.clone()
            
            self.fifo_counter = self.fifo_counter + x.shape[0]
           
     def get_fifo_data(self):
                
            X, Y = self.fifo_x.clone(), self.fifo_y.clone()
            
            support_x, support_y = [], []
    
            unique_labels = torch.arange(self.classes_per_task) 
            
            # samples per label
            for label in unique_labels:
                matched_ids = (Y == label).nonzero(as_tuple=True)[0]
                
                # handle edge case (less than k samples)
                num_samples = min(self.fifo_samples_per_label, matched_ids.size(0))
                
                rand_ids = matched_ids[torch.randperm(matched_ids.size(0))[:num_samples]]
                
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
        
  
   
     



