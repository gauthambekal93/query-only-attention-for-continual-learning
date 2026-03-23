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
from torch.utils.data import DataLoader

class DataManager:
     
     def __init__(self, root, data_dir, num_images_per_class, initial_num_classes, total_classes, num_labels_previous_task, num_data_points_current_task, num_support_per_label, task_lag, device):
         
         self.device = device
         self.initial_num_classes = initial_num_classes
         self.total_classes = total_classes
         self.current_task_id = 0
         
         #self.num_label_rows  = 5
         
         #self.num_train_classes = initial_num_classes * ( self.num_label_rows + 1 )
         
         self.total_tasks = int( self.total_classes / self.initial_num_classes )
         
         self.num_images_per_task = num_images_per_class * initial_num_classes
         
         self.data_path = os.path.join( root, data_dir)
         
         self.pad = 4 
         
         self.num_data_points_current_task =num_data_points_current_task
         
         self.num_support_per_label = num_support_per_label
         
         self.task_lag = task_lag
         
         self.train_x, self.train_y,  self.test_x, self.test_y  = [] , [] , [], [] 
         
         self.task_train_x, self.task_mapped_train_y , self.task_unmapped_train_y = {}, {}, {}
         
         self.task_val_x, self.task_mapped_val_y, self.task_unmapped_val_y = {}, {}, {}
         
         self.task_test_x, self.task_mapped_test_y, self.task_unmapped_test_y =  {}, {}, {}
         
         self.validation_accuracy = []
         
         self.local_eval_support_x, self.local_eval_support_y = {}, {}
         
         self.global_data = { i: torch.empty( 0 ) for i in range(self.total_classes)}
         
         self.global_eval_support_x = None
        
  
        
     def create_tiny_imagenet_data(self):
      
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        
        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             normalize, ])
        transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
        
        self.train_set = datasets.ImageFolder(root=os.path.join(self.data_path, 'train'), transform=transform_train)
        
        self.test_set = datasets.ImageFolder(root=os.path.join(self.data_path, 'val'), transform=transform_test)

        
        assert self.train_set.class_to_idx == self.test_set.class_to_idx, "DATA LOADING WENT WRONG"
       
        loader = DataLoader(self.train_set, batch_size=1000, shuffle=False,num_workers=1, pin_memory=True )
        
        count =0
        for img, label in loader:
            print(count)
            self.train_x.append(img)
            self.train_y.append(label)
            count +=1
        
            
        self.train_x = torch.cat (self.train_x, dim =0)
        
        self.train_y = torch.cat(self.train_y, dim =0)
        
        loader = DataLoader(self.test_set, batch_size=1000, shuffle=False,num_workers=1, pin_memory=True )
        
        count =0
        for img, label in loader:
            print(count)
            self.test_x.append(img)
            self.test_y.append(label)
            count +=1
            
            
        self.test_x =torch.cat (self.test_x, dim =0)
        
        self.test_y = torch.cat(self.test_y, dim =0)
        
        save_path = os.path.join(self.data_path, "tiny_imagenet_cached.pt")

        torch.save(
            {
                "train_x": self.train_x.cpu(),
                "train_y": self.train_y.cpu(),
                "test_x":  self.test_x.cpu(),
                "test_y":  self.test_y.cpu(),
                "class_to_idx": self.train_set.class_to_idx,
            },
            save_path
        )
        
        print(f"Saved Tiny ImageNet cache to {save_path}")

     
     def load_tiny_imagenet_data(self):
         
         load_path = os.path.join(self.data_path, "tiny_imagenet_cached.pt")

         data = torch.load(load_path, map_location="cpu")
        
         self.train_x = data["train_x"].to(self.device)
         self.train_y = data["train_y"].to(self.device)
        
         self.test_x  = data["test_x"].to(self.device)
         self.test_y  = data["test_y"].to(self.device)
        
         self.class_to_idx = data["class_to_idx"]
        
         print("Loaded Tiny ImageNet cache")

     
        
     def label_remapping(self, labels ):
         
         classes = torch.full((self.total_classes,), -1, device=self.device) 
         
         classes[self.unique_task_labels] = torch.arange(len(self.unique_task_labels), device = self.device)

         # Map labels → indices
         index = classes[labels]
         
         return index

     
     def get_train_support(self, task_id, corrupt_labels: bool = False):  
             #the bool is set to True only for corrupting the support and seeing for drop in accuracy.
             support_x, support_y = [], []
         
             for label in torch.unique(self.task_mapped_train_y[task_id]):
                 
                 support_indices = (self.task_mapped_train_y[task_id] == label).nonzero(as_tuple=True)[0]
                 
                 idx = torch.randperm(support_indices.size(0))[: self.num_support_per_label]
                 
                 support_indices = support_indices[idx]
         
                 support_x.append(self.task_train_x[task_id][support_indices])
                 
                 support_y.append(self.task_mapped_train_y[task_id][support_indices])
         
             support_x = torch.cat(support_x, dim=0)
             
             support_y = torch.cat(support_y, dim=0)
         
             support_y = F.one_hot(support_y, num_classes=self.initial_num_classes).to(self.device, dtype=torch.float32)
         
             # ---- corruption: break x↔y alignment (shuffle labels only) ----
             if corrupt_labels:
                 
                 perm = torch.randperm(support_y.size(0), device=support_y.device)
                 
                 support_y = support_y[perm]   # <-- DO NOT permute support_x

             return support_x, support_y
         
            
     def create_eval_support(self):
         
         support_x, support_y = [], []
     
         for label in torch.unique(self.task_mapped_train_y[self.current_task_id]):
             
             support_indices = (self.task_mapped_train_y[self.current_task_id] == label).nonzero(as_tuple=True)[0]
             
             idx = torch.randperm(support_indices.size(0))[: self.num_support_per_label]

             support_indices = support_indices[idx]
     
             support_x.append(self.task_train_x[self.current_task_id][support_indices])
             
             support_y.append(self.task_mapped_train_y[self.current_task_id][support_indices])
     
         self.local_eval_support_x[self.current_task_id]  = torch.cat(support_x, dim=0)
         
         support_y = torch.cat(support_y, dim=0)
        
         self.local_eval_support_y[self.current_task_id] = F.one_hot(support_y, num_classes=self.initial_num_classes).to(self.device, dtype=torch.float32)
      
        

     def create_global_data(self, X, Y):
         
         for label in self.unique_task_labels:
             
             mask = torch.isin( Y , label)
             
             if len(self.global_data[label.item()])==0:  
                     
                     self.global_data[label.item()] = X[mask][:20]
                     
                  
                     
     def get_global_supports(self):
         
         label_history = torch.unique( torch.cat(list(self.task_unmapped_train_y.values()), dim=0) )
         
         global_eval_support_x = torch.cat( [ img.to(self.device) for label, img in self.global_data.items() if label in label_history ] , dim = 0)
    
         global_eval_support_y = torch.cat( [ torch.tensor( [label] * self.num_support_per_label ) for label in self.global_data.keys() if label in label_history ],dim = 0)
         
         global_eval_support_y = F.one_hot( global_eval_support_y, num_classes = self.total_classes  ).to(self.device)  
         
         return global_eval_support_x, global_eval_support_y, label_history
         
                
         
     def create_task_data(self):
         
             """Choose Random Labels for the task"""
             self.unique_task_labels = torch.randperm(self.total_classes)[: self.initial_num_classes ].to(self.device)
            
            
             """Create train and validation splits"""
             train_mask = torch.isin( self.train_y,  self.unique_task_labels)
            
             rand_ids = torch.randperm(self.train_y [ train_mask ] .shape[0])
        
             train_val_ratio = 0.80
            
             train_rand_ids = rand_ids[: int( len(rand_ids) * train_val_ratio ) ]
            
             val_rand_ids= rand_ids [ int( len(rand_ids) * train_val_ratio ) : ]
            
            
             """Create train data """
             self.task_train_x[self.current_task_id] = self.augment_batch ( self.train_x[train_mask][train_rand_ids] )
             
             self.task_unmapped_train_y[self.current_task_id] = self.train_y [ train_mask ][train_rand_ids]  
             
             self.task_mapped_train_y[self.current_task_id] = self.label_remapping(self.task_unmapped_train_y[self.current_task_id]  )
             
             
             """Create val data """
             self.task_val_x[self.current_task_id] = self.train_x [train_mask][val_rand_ids]
             
             self.task_unmapped_val_y[self.current_task_id] = self.train_y [ train_mask ][val_rand_ids]
             
             self.task_mapped_val_y[self.current_task_id] = self.label_remapping( self.task_unmapped_val_y[self.current_task_id]   ) 
                   
                    
             """Create test data """
             test_mask = torch.isin( self.test_y,  self.unique_task_labels)
            
             self.task_test_x[self.current_task_id] = self.test_x [test_mask]
             
             self.task_unmapped_test_y[self.current_task_id] =  self.test_y [ test_mask ]
             
             self.task_mapped_test_y[self.current_task_id] = self.label_remapping( self.task_unmapped_test_y[self.current_task_id] )
             
             
             """Create support data for task level evaluvation """
             self.create_eval_support()
             
             
             """Create support data for global evaluvation """
             self.is_filled = all( [ True if len(imgs) >=self.num_support_per_label else False for imgs in self.global_data.values()  ] )
             
             if self.is_filled is False:
                 
                 self.create_global_data( self.task_train_x[self.current_task_id] , self.task_unmapped_train_y[self.current_task_id]  )
            
                     
                     
        
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - self.task_lag]
         
         del self.task_unmapped_train_y[self.current_task_id - self.task_lag]
         
         del self.task_mapped_train_y[self.current_task_id - self.task_lag]
         
         
         del self.task_val_x[self.current_task_id - self.task_lag]
         
         del self.task_unmapped_val_y[self.current_task_id - self.task_lag]
         
         del self.task_mapped_val_y[self.current_task_id - self.task_lag]
         
         
         del self.task_test_x[self.current_task_id - self.task_lag]
         
         del self.task_unmapped_test_y[self.current_task_id - self.task_lag]
        
         del self.task_mapped_test_y[self.current_task_id - self.task_lag]
        
        
         del self.local_eval_support_x[self.current_task_id - self.task_lag]
         
         del self.local_eval_support_y[self.current_task_id - self.task_lag]



        
     
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

     
'''     
for label, img in self.global_data.items():
    if label in label_history:
        print(label, img.shape)

'''

'''
for k, imgs in self.global_data.items() :
    print(k, len(imgs)) 
'''    
    
    