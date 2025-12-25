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

class DataManager:
     
     def __init__(self, root, data_dir, num_images_per_class, initial_num_classes, class_increase_per_task, total_classes, device):
         
         self.current_num_classes = initial_num_classes
         self.class_increase_per_task = class_increase_per_task
         self.total_classes = total_classes
         self.current_task_id = 0
         
         self.total_tasks = int( self.total_classes / self.class_increase_per_task )
         
         self.num_images_per_task = num_images_per_class * class_increase_per_task
         
         self.data_path = os.path.join( root, data_dir)
         
         """we are assigning 5 labels per task, hence 100 labels in cifar creates 20 tasks """
         self.label_ids  = torch.randperm(total_classes)
         
         #self.label_ids_flattened = self.label_ids.clone() #self.label_ids.copy()
         
         self.label_ids = self.label_ids.reshape(-1, class_increase_per_task)
         
         """ The below 4 varibales will contain entire cfiar dataset and be used to create task specific data"""
         self.comp_train_x , self.comp_train_y , self.comp_test_x,  self.comp_test_y = {}, {}, {}, {}
         
         self.device = device
         
         self.pad = 4 
         
     def create_cifar_data(self):
        
        """The numbers are mean and std across 3 channels of the image.
            I have confirmed these mean and std values are correct, 
            by initailly downloading and manually inspecting meand and std"""
     
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))   
        ])
        
        train_set = datasets.CIFAR100(
            root=self.data_path ,
            train=True,
            download=False,  
            transform=transform
        )
     
        test_set = datasets.CIFAR100(
            root=self.data_path ,
            train=False,
            download=False,  
            transform=transform
        )
     
     
        """ We are assigning random labels to one of the possible 20 tasks. 
        Each task will contain 5 labels and all 100 labels assigned to 20 tasks
        """
        for img, label in train_set:
            task_id = (self.label_ids == label).nonzero(as_tuple=True)[0].item()  #np.where(self.label_ids== label)[0][0]
            
            if task_id not in self.comp_train_y:
                self.comp_train_x[task_id] , self.comp_train_y[task_id] = [], []
            
            self.comp_train_x[task_id].append(img)
            
            self.comp_train_y[task_id].append(label)
     
            
        for img, label in test_set:
             task_id = (self.label_ids == label).nonzero(as_tuple=True)[0].item() #np.where(self.label_ids== label)[0][0]
             
             if task_id not in self.comp_test_y:
                 self.comp_test_x[task_id] , self.comp_test_y[task_id] = [], []
             
             self.comp_test_x[task_id].append(img)
             
             self.comp_test_y[task_id].append(label)       
     
        
        """ Store data of each task as torch tensor from list """
        for k in self.comp_train_x.keys():
            self.comp_train_x[k] = torch.stack(self.comp_train_x [k], dim = 0)
            self.comp_train_y[k] = torch.tensor(self.comp_train_y [k])
            self.comp_test_x[k] = torch.stack(self.comp_test_x [k], dim = 0)
            self.comp_test_y[k] = torch.tensor(self.comp_test_y [k])
            
            
     def get_one_hot_encoded(self, labels):

        index = [] 
        for label in labels:
            
            index .append( (self.label_ids[:self.current_task_id + 1].reshape(-1) == label).nonzero(as_tuple=True)[0].item() )
            
        one_hot_encoded = F.one_hot( torch.tensor(index), num_classes =  self.current_num_classes)
        
        return one_hot_encoded
        
     
     def create_task_data(self):
         
         self.task_train_x = torch.cat( [ self.comp_train_x[task_id] for task_id in range(self.current_task_id + 1) ] ).to(self.device)
         self.task_train_y = self.get_one_hot_encoded( torch.cat( [ self.comp_train_y[task_id] for task_id in range(self.current_task_id + 1)])).to(self.device, dtype=torch.float32)  
             
         self.task_test_x = torch.cat( [ self.comp_test_x[task_id] for task_id in range(self.current_task_id + 1) ] ).to(self.device)
         self.task_test_y = self.get_one_hot_encoded(  torch.cat( [ self.comp_test_y[task_id] for task_id in range(self.current_task_id + 1) ])).to(self.device, dtype=torch.float32)  
         
         self.selected_classes = self.label_ids[:self.current_task_id + 1].reshape(-1).to(self.device)
        


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



