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
         
         """we are assigning 5 labels per task, hence 100 labels in cifar creates 20 tasks """
         self.label_ids  = torch.randperm(total_classes)#.to(device)
         
         
         if class_increase_per_task>1:
             self.label_ids = self.label_ids.reshape(-1, class_increase_per_task).to (self.device)
         
         
            
         """ The below 6 varibales will contain entire cfiar dataset and be used to create task specific data"""
         
         #train_data_x = { task_id: [] for task_id in range(self.total_tasks) }
         #train_data_y = { task_id: [] for task_id in range(self.total_tasks) }
         
         #self.comp_val_x = { task_id: [] for task_id in range(self.total_tasks) }
         #self.comp_val_y = { task_id: [] for task_id in range(self.total_tasks) }
         
         self.comp_test_x = { task_id: [] for task_id in range(self.total_tasks) }
         self.comp_test_y = { task_id: [] for task_id in range(self.total_tasks) }

             
         self.pad = 4 
         
         self.test_support_x = {}
         
         self.test_support_y = {}
         
         #self.num_labels_previous_task = num_labels_previous_task
         
         self.num_data_points_current_task =num_data_points_current_task
         
         self.num_support_per_label = num_support_per_label
         
         self.task_train_x, self.task_train_y, self.task_val_x, self.task_val_y  = {}, {}, {}, {}
         
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
     
     
        """ We are assigning random labels to one of the possible 20 tasks. 
        Each task will contain 5 labels and all 100 labels assigned to 20 tasks
        """
        
        
        #base_labels = self.label_ids[0]
        base_labels = self.label_ids[-1]
        
        train_data = { label.item(): [] for label in base_labels }
        
        for img, label in self.train_set:
            if label in base_labels:
                train_data[label].append(img)
        
        """Convert list of images to a image tensors """
        for label, img in train_data.items():
            train_data[label] = torch.stack( train_data[label], dim = 0)
            
        """ Create newer labels and images to be created from base labels"""   
        #labels = self.label_ids[: self.num_label_rows + 1 ].reshape(-1).to(self.device) 
        labels = self.label_ids[ - self.num_label_rows - 1 :].reshape(-1).to(self.device) 
       
        mask = ~torch.isin(labels, base_labels)
        
        addition_labels = labels[mask]
         
        for label in addition_labels:
            
            rand_idx = torch.randperm( len(base_labels))[0]
            
            base_label = base_labels[rand_idx].item()
            
            pixel_permutation = torch.randperm( train_data[base_label].shape[2])  
            
            images = train_data[base_label]
                
            permuted_images = images[:,:, pixel_permutation, :]
            
            permuted_images = permuted_images[:,:,:, pixel_permutation]
            
            train_data[label.item()]  = permuted_images
        
        
        '''
        base_labels = self.label_ids[: self.num_label_rows + 1 ].reshape(-1)
        
        train_data = { label.item(): [] for label in base_labels }
        
        for img, label in self.train_set:
            if label in base_labels:
                train_data[label].append(img)
        
        """Convert list of images to a image tensors """
        for label, img in train_data.items():
            train_data[label] = torch.stack( train_data[label], dim = 0)
        '''
            
            
        train_size_per_label , val_size_per_label = 400, 100
        
        """Split data train and validation for the task seperately"""
        self.task_train_x[self.current_task_id] = torch.cat([images[:train_size_per_label] for images in train_data.values()]).to(self.device)
        
        self.task_val_x[self.current_task_id] = torch.cat([images[ - val_size_per_label :] for images in train_data.values()]).to(self.device)
        
        self.task_train_y[self.current_task_id] = torch.cat([ torch.tensor([label]*train_size_per_label) for label in train_data.keys()]).to(self.device)
        
        self.task_train_y[self.current_task_id] = self.label_remapping( self.task_train_y[self.current_task_id] )
        
        self.task_val_y[self.current_task_id] = torch.cat([ torch.tensor([label]*val_size_per_label) for label in train_data.keys()]).to(self.device)
        
        self.task_val_y[self.current_task_id] = self.label_remapping(self.task_val_y[self.current_task_id] )
      
        
        """Carry out data permuation on train data """
        #data_permutation =  torch.randperm( self.task_train_x[self.current_task_id ].shape[0] )
        
        #self.task_train_x[self.current_task_id] = self.task_train_x[self.current_task_id ][data_permutation, :, :, :]   
             
        #self.task_train_y[self.current_task_id ] = self.task_train_y[self.current_task_id ][data_permutation]
             
     
              
     
     def label_remapping(self, labels ):
          
         index = []
         #label_ids = self.label_ids[: self.num_label_rows + 1].reshape(-1)
         label_ids = self.label_ids[ - self.num_label_rows - 1 : ].reshape(-1)
         for label in labels:
             
             index .append( (label == label_ids).nonzero(as_tuple=True)[0].item() )
          
         index = torch.tensor(index).to(self.device)
         
         return index

    
     def label_remapping_test(self, labels, task_id ):
          
         label_ids = self.label_ids[: task_id + 1].reshape(-1)
         
         mapping = torch.full((self.total_classes,), -1, device=label_ids.device, dtype=torch.long)
        
         mapping[label_ids] = torch.arange(label_ids.numel(), device=label_ids.device)

         new_labels = mapping[labels]   
          
         return new_labels
 
    
     
     def create_task_data(self):
             
             """Obtain pixel and data permutation """
             pixel_permutation = torch.randperm(self.task_train_x[self.current_task_id - 1].shape[2])  
             
             data_permutation =  torch.randperm( self.task_train_x[self.current_task_id - 1].shape[0] )
             
             
             """Carry out pixel and data permuation on train data """
             self.task_train_x[self.current_task_id] = self.task_train_x[self.current_task_id - 1][data_permutation, :, :, :]   
             
             self.task_train_y[self.current_task_id ] = self.task_train_y[self.current_task_id - 1][data_permutation]
             
             self.task_train_x[self.current_task_id] = self.task_train_x[self.current_task_id ][:, :, pixel_permutation , :]
             
             self.task_train_x[self.current_task_id] =  self.task_train_x[self.current_task_id][:, :, :, pixel_permutation]
             
             
             """Carry out pixel and data permuation on validation data """
             self.task_val_x[self.current_task_id] = self.task_val_x[self.current_task_id -1][:, :, pixel_permutation , : ]
            
             self.task_val_x[self.current_task_id] = self.task_val_x[self.current_task_id][:, :, :, pixel_permutation]
            
             self.task_val_y[self.current_task_id] =  self.task_val_y[self.current_task_id -1]
     '''
       
     def create_task_data(self):
             
             """Obtain pixel and data permutation """
             #pixel_permutation = torch.randperm(self.task_train_x[self.current_task_id - 1].shape[2])  
             
             #data_permutation =  torch.randperm( self.task_train_x[self.current_task_id - 1].shape[0] )
             
             
             """Carry out pixel and data permuation on train data """
             self.task_train_x[self.current_task_id] = self.task_train_x[self.current_task_id - 1][:, :, :, :]   
             
             self.task_train_y[self.current_task_id ] = self.task_train_y[self.current_task_id - 1][:]
             
             self.task_train_x[self.current_task_id] = self.task_train_x[self.current_task_id ][:, :, : , :]
             
             self.task_train_x[self.current_task_id] =  self.task_train_x[self.current_task_id][:, :, :, :]
             
             
             """Carry out pixel and data permuation on validation data """
             self.task_val_x[self.current_task_id] = self.task_val_x[self.current_task_id -1][:, :, : , : ]
            
             self.task_val_x[self.current_task_id] = self.task_val_x[self.current_task_id][:, :, :, :]
            
             self.task_val_y[self.current_task_id] =  self.task_val_y[self.current_task_id -1]
     '''
     
     
     def delete_data(self):
         
         del self.task_train_x[self.current_task_id - 20]
         
         del self.task_train_y[self.current_task_id - 20] 
         
         del self.task_val_x[self.current_task_id - 20]
         
         del self.task_val_y[self.current_task_id - 20]
         
        
        
     def get_support(self, task_id):
         
         support_x, support_y = [], []
         
         for label in torch.unique(self.task_train_y[task_id ]) :
             
             support_indices = (self.task_train_y[task_id ] == label) .nonzero(as_tuple=True)[0]
             
             idx = torch.randperm(support_indices.size(0)) [: self.num_support_per_label ]
             
             support_indices = support_indices[idx]
            
             support_x.append(  self.task_train_x [task_id][support_indices]  )
             
             support_y.append( self.task_train_y [task_id][support_indices] )
         
         support_x = torch.cat(support_x , dim =0)
            
         support_y = torch.cat(support_y , dim =0)
            
         support_y = F.one_hot( support_y  , num_classes = self.num_train_classes ).to(self.device, dtype=torch.float32) 
         
         return support_x, support_y
     
     
     
     
     def create_test_data(self):
         
        self.cifar_test_x = {task_id : [] for task_id in range(self.total_tasks)} 
        self.cifar_test_y = {task_id : [] for task_id in range(self.total_tasks)} 
        self.cifar_train_x = {task_id : [] for task_id in range(self.total_tasks)} 
        self.cifar_train_y = {task_id : [] for task_id in range(self.total_tasks)} 
        
        x, y = [], []    
    
        for img, label in self.test_set:
            x.append(img)
            y.append(label)
                
        x = torch.stack( x  , dim = 0).to(self.device)
        
        y = torch.tensor( y ).to(self.device)
        
        
        for task_id in range(self.total_tasks):
            
            labels = self.label_ids[ : task_id + 1 ].reshape(-1)
            
            mask = torch.isin(y, labels)
            
            indices = torch.nonzero(mask).squeeze()
      
            self.cifar_test_x[task_id] = x[indices]
            
            self.cifar_test_y[task_id] = self.label_remapping_test( y[indices], task_id )
            
            
        x, y = [], []    
    
        for img, label in self.train_set:
            x.append(img)
            y.append(label)
                
        x = torch.stack( x  , dim = 0).to(self.device)
        
        y = torch.tensor( y ).to(self.device)
        
        for task_id in range(self.total_tasks):
            
            labels = self.label_ids[ : task_id + 1 ].reshape(-1)
            
            mask = torch.isin(y, labels)
            
            indices = torch.nonzero(mask).squeeze()
      
            self.cifar_train_x[task_id]  = x[indices]
            
            self.cifar_train_y[task_id] = self.label_remapping_test( y[indices], task_id )
            
      
            
     def get_test_support(self, task_id):
                 
         support_x, support_y = [], []
         
         #test_classes = len( torch.unique(self.cifar_train_y[task_id]))
         
         for label in torch.unique(self.cifar_train_y[task_id]) :
             
             support_indices = (self.cifar_train_y[task_id] == label) .nonzero(as_tuple=True)[0] [: self.num_support_per_label ]
             
             support_x.append(  self.cifar_train_x[task_id][support_indices]  )
             
             support_y.append( self.cifar_train_y[task_id] [support_indices] )
         
         support_x = torch.cat(support_x , dim =0)
            
         #support_y = torch.cat(support_y , dim =0)
            
         #support_y = F.one_hot( support_y  , num_classes = test_classes ).to(self.device, dtype=torch.float32) 
         
         return support_x
     
        

     
     
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

     
     
         

         


