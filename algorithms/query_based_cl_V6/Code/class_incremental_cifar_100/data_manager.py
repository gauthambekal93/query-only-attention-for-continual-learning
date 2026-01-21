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
         
         self.current_num_classes = initial_num_classes
         self.class_increase_per_task = class_increase_per_task
         self.total_classes = total_classes
         self.current_task_id = 0
         
         self.total_tasks = int( self.total_classes / self.class_increase_per_task )
         
         self.num_images_per_task = num_images_per_class * class_increase_per_task
         
         self.data_path = os.path.join( root, data_dir)
         
         """we are assigning 5 labels per task, hence 100 labels in cifar creates 20 tasks """
         self.label_ids  = torch.randperm(total_classes)#.to(device)
         
         
         if class_increase_per_task>1:
             self.label_ids = self.label_ids.reshape(-1, class_increase_per_task)
         
            
         """ The below 6 varibales will contain entire cfiar dataset and be used to create task specific data"""
         
         self.comp_train_x = { task_id: [] for task_id in range(self.total_tasks) }
         self.comp_train_y = { task_id: [] for task_id in range(self.total_tasks) }
         
         self.comp_val_x = { task_id: [] for task_id in range(self.total_tasks) }
         self.comp_val_y = { task_id: [] for task_id in range(self.total_tasks) }
         
         self.comp_test_x = { task_id: [] for task_id in range(self.total_tasks) }
         self.comp_test_y = { task_id: [] for task_id in range(self.total_tasks) }

             
         self.pad = 4 
         
         self.test_support_x = {}
         
         self.test_support_y = {}
         
         self.num_labels_previous_task = num_labels_previous_task
         
         self.num_data_points_current_task = num_data_points_current_task
         
         self.num_support_per_label = num_support_per_label
         
         self.replay_train_x, self.replay_train_y, self.replay_val_x, self.replay_val_y = None, None, None, None
         
         
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
            task_id = (self.label_ids == label).nonzero(as_tuple=True)[0].item()  
            
            if len(self.comp_train_x[task_id]) < self.num_data_points_current_task:
                self.comp_train_x[task_id].append(img)
                
                self.comp_train_y[task_id].append(label)
            else:
                self.comp_val_x[task_id].append(img)
                
                self.comp_val_y[task_id].append(label)
            
            
        for img, label in test_set:
             task_id = (self.label_ids == label).nonzero(as_tuple=True)[0].item() 
             
             self.comp_test_x[task_id].append(img)
             
             self.comp_test_y[task_id].append(label)       
     
        
        """ Store data of each task as torch tensor from list """
        for k in self.comp_train_x.keys():
            
            self.comp_train_x[k] = torch.stack(self.comp_train_x [k], dim = 0)
            self.comp_train_y[k] = torch.tensor(self.comp_train_y [k])
            
            self.comp_val_x[k] = torch.stack(self.comp_val_x [k], dim = 0)
            self.comp_val_y[k] = torch.tensor(self.comp_val_y [k])
            
            self.comp_test_x[k] = torch.stack(self.comp_test_x [k], dim = 0)
            self.comp_test_y[k] = torch.tensor(self.comp_test_y [k])
            

     
     def label_remapping(self, labels):
     
         index = []
         
         for label in labels:
             
             index .append( (self.selected_classes == label).nonzero(as_tuple=True)[0].item() )
          
         index = torch.tensor(index).to(self.device)
         
         return index
     
     
     def fill_replay_buffer(self):
         
            for label in self.selected_classes:
                
                sample_ids = (  self.unmapped_train_y == label).nonzero(as_tuple=True)[0][: self.num_labels_previous_task]
                
                if self.replay_train_x is None:

                    self.replay_train_x = self.task_train_x[sample_ids]
                
                    self.replay_train_y = self.unmapped_train_y[sample_ids]
                
                else:
                     self.replay_train_x = torch.cat( (self.replay_train_x, self.task_train_x[sample_ids]), dim = 0 )
                 
                     self.replay_train_y = torch.cat( (self.replay_train_y, self.unmapped_train_y[sample_ids]), dim = 0 )
                 
 
     def create_train_data(self):
         
         self.selected_classes = self.label_ids[:self.current_task_id + 1].reshape(-1).to(self.device)
         
         self.task_train_x = self.comp_train_x[self.current_task_id].to(self.device) 
         self.unmapped_train_y = self.comp_train_y[self.current_task_id].to(self.device)
         self.task_train_y = self.label_remapping( self.unmapped_train_y )
         
        
         if self.replay_train_x is not None:
             self.task_train_x = torch.cat( (self.task_train_x, self.replay_train_x), dim = 0)
             self.unmapped_train_y = torch.cat( (self.unmapped_train_y, self.replay_train_y), dim = 0)
         
         
        
     def create_eval_data(self):
           
           self.task_val_x, self.task_val_y , self.task_test_x, self.task_test_y = {}, {}, {}, {}
           
           for task_id in range(self.current_task_id + 1):
               
               self.task_val_x[task_id] =  self.comp_val_x[task_id].to(self.device)
               
               self.task_val_y[task_id] =  self.label_remapping( self.comp_val_y[task_id]).to(self.device)
               
               self.task_test_x[task_id] = self.comp_test_x[task_id].to(self.device) 
               
               self.task_test_y[task_id] = self.label_remapping( self.comp_test_y[task_id]).to(self.device)   
           
 
     def create_prediction_support(self):
         
         self.support_x = {}
         
         for label in self.selected_classes:
             
             indices = (  self.unmapped_train_y == label).nonzero(as_tuple=True)[0][:1]
             
             """We map the label """
             label =  (self.selected_classes == label).nonzero(as_tuple=True)[0].item() 
             
             self.support_x[label] = self.task_train_x [indices]
             
             
     def get_binary_support(self, unmapped_y):
           
           matched_indices, mismatched_indices = {}, {}
           
           for label in torch.unique( unmapped_y ):
               
               matched_indices[label.item()] = (  self.unmapped_train_y == label).nonzero(as_tuple=True)[0]
               
               mismatched_indices[label.item()] = (  self.unmapped_train_y != label).nonzero(as_tuple=True)[0]
               
           support_x, binary_labels = [], []
           
           for i, label in enumerate( unmapped_y ):
               
               label = label.item()
                
               if (torch.rand(1) < 0.5).int():
                   rand_idx = torch.randperm( len( matched_indices[label]))[:1]
                   support_indices = matched_indices[label][rand_idx]
                   binary_labels.append(1)
               else:
                   rand_idx = torch.randperm( len( mismatched_indices[label]))[:1]
                   support_indices = mismatched_indices[label][rand_idx]
                   binary_labels.append(0)
              
               support_x.append( self.augment_batch ( self.task_train_x [support_indices] ) )
               
           support_x = torch.cat(support_x , dim = 0)

           binary_labels = torch.tensor(binary_labels, dtype = torch.float32 ).to(self.device)
             
           return support_x, binary_labels
       
      
        
     def get_eval_support(self):

         support_x, support_y = [], []
         
         for label in self.selected_classes:
           
             support_indices = (  self.replay_train_y == label).nonzero(as_tuple=True)[0] [: self.num_support_per_label ]
             
             support_x.append(  self.replay_train_x [support_indices]  )  
             
             support_y.append( self.replay_train_y [support_indices] )
             
         support_x = torch.cat(support_x , dim =0)
         
         support_y = torch.cat(support_y , dim =0)
         
         support_y = F.one_hot( support_y  , num_classes = self.total_classes ).to(self.device, dtype=torch.float32) 
         
         return support_x, support_y
             
     
        
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
         def get_train_support(self, unmapped_y):

             """Here we want both matching and not matching labels to be part of support for each input labels.
                We also want different -ve label supports for a given query label for different datapoints to increase variablity.
                eg. query label 10 can have -ve support label(2,5,8,9) and  query label 10 can have -ve support label(3,5,4,1) """
             
             matched_indices, mismatched_indices = {}, {}
             
             support_x, support_y = [], []
             
             for label in torch.unique( unmapped_y ):
                 
                 matched_indices[label.item()] = (  self.unmapped_train_y == label).nonzero(as_tuple=True)[0]
                 
                 mismatched_indices[label.item()] = (  self.unmapped_train_y != label).nonzero(as_tuple=True)[0]
                 
             support_y_template = torch.tensor(([1,0], [0,1])).to( self.device )
            
            
             for i, label in enumerate( unmapped_y ):
                 
                 label = label.item()
                 
                 rand_idx = torch.randperm( len( matched_indices[label]))[:1]
                 
                 sampled_matching_indices = matched_indices[label][rand_idx]
                 
                 rand_idx = torch.randperm( len( mismatched_indices[label]))[:1]
                 
                 sampled_mismatching_indices = mismatched_indices[label][rand_idx]
                
                 support_indices = torch.cat( (sampled_matching_indices, sampled_mismatching_indices), dim = 0 )
                 
                 rand_idx = torch.randperm(len(support_indices))
                 
                 support_indices = support_indices[rand_idx]
                 
                 support_x.append( self.augment_batch ( self.task_train_x [support_indices] ) )
                 
                 support_y.append( support_y_template[rand_idx] )

             support_x = torch.cat(support_x , dim =0)
              
             support_y = torch.cat(support_y , dim =0)
              
             return support_x, support_y
          
         '''
          
         

         


