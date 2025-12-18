# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 08:22:19 2025

@author: gauthambekal93
"""
import os
#import json
import numpy as np
from torchvision import datasets, transforms
#from pathlib import Path
#import argparse
#import sys
import torch
import pickle

class CifarData:
    
    def __init__(self, ROOT, data_params, model_params):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))   # CIFAR-100 mean & std
        ])
        
        self.current_task_id = 0
        self.model_params = model_params
        self.data_params = data_params
       
        self.ROOT = ROOT
        
        self.result_path = os.path.join( self.ROOT, self.model_params["model_dir"], 'results.pkl' )
        
        self.model_path = os.path.join(self.ROOT, self.model_params["model_dir"], 'model.pth' )
        
        self.current_running_avg_step = 0
        
        total_tasks = int( data_params["total_classes"]/data_params["class_increase_per_task"] )
        epochs_per_task = int(model_params['num_epochs'] / (total_tasks)) 
        
        bin_size = int( (data_params["running_avg_window"] * model_params["batch_sizes"]["train"]))
        
        num_images_per_task = data_params["num_images_per_class"] * data_params["class_increase_per_task"]
        
        total_updates = np.sum([(i + 1) * num_images_per_task * epochs_per_task for i in range(total_tasks)])
            
        self.results_dict = {}
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(  int(total_updates/ bin_size) )
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros(  int(total_updates/ bin_size) )
        
        self.results_dict["test_loss_per_epoch"] = torch.zeros(  model_params["num_epochs"] )
        self.results_dict["test_accuracy_per_epoch"] =  torch.zeros(  model_params["num_epochs"] )
        
        
        
    def create_cifar_data(self):
       
       data_path = os.path.join( self.ROOT, self.data_params["data_dir"])
       
       
       """The numbers are mean and std across 3 channels of the image.
          I have confirmed these mean and std values are correct, 
          by initailly downloading and manually inspecting meand and std"""
    
       
       train_set = datasets.CIFAR100(
           root=data_path,
           train=True,
           download=False,  
           transform=self.transform
       )
    
       test_set = datasets.CIFAR100(
           root=data_path,
           train=False,
           download=False,  
           transform=self.transform
       )
    
       """we are assigning 5 labels per task, hence 100 labels in cifar creates 20 tasks """
    
       label_ids = np.random.permutation(self.data_params["total_classes"])
       
       label_ids = label_ids.reshape(-1, self.data_params["class_increase_per_task"])
    
       self.comp_train_x , self.comp_train_y= {}, {}
    
       self.comp_test_x , self.comp_test_y= {}, {}
    
       """ We are assigning random labels to one of the possible 20 tasks. 
       Each task will contain 5 labels and all 100 labels assigned to 20 tasks
       """
       for img, label in train_set:
           task_id = np.where(label_ids== label)[0][0]
           
           if task_id not in self.comp_train_y:
               self.comp_train_x[task_id] , self.comp_train_y[task_id] = [], []
           
           self.comp_train_x[task_id].append(img)
           
           self.comp_train_y[task_id].append(label)
    
           
    
       for img, label in test_set:
            task_id = np.where(label_ids== label)[0][0]
            
            if task_id not in self.comp_test_y:
                self.comp_test_x[task_id] , self.comp_test_y[task_id] = [], []
            
            self.comp_test_x[task_id].append(img)
            
            self.comp_test_y[task_id].append(label)       
       
       
    def create_task_data(self):
        
        if self.current_task_id ==0:
            
            self.task_train_x =  torch.stack( self.comp_train_x[self.current_task_id], dim = 0) 
            
            self.task_train_y =   torch.tensor(self.comp_train_y[self.current_task_id])
            
            self.task_test_x = torch.stack(  self.comp_test_x[self.current_task_id], dim = 0 )
            
            self.task_test_y =   torch.tensor ( self.comp_test_y[self.current_task_id] )
            
        else:
            self.task_train_x = torch.cat ( self.task_train_x,  torch.stack( self.comp_train_x[self.current_task_id], dim = 0) , dim = 0)
            
            self.task_train_y = torch.cat ( self.task_train_y,  torch.tensor(self.comp_train_y[self.current_task_id]), dim = 0 )
            
            self.task_test_x = torch.cat ( self.task_test_x, torch.stack(  self.comp_test_x[self.current_task_id], dim = 0 ), dim = 0)
            
            self.task_test_y = torch.cat ( self.task_test_y,  torch.tensor ( self.comp_test_y[self.current_task_id] ), dim = 0 )
    
    
    def create_result_dir(self):
        
        os.makedirs( os.path.join(self.ROOT, self.model_params["model_dir"]), exist_ok=True)
    

    def save_train(self, current_accuracy, current_reg_loss, net):
        
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] =  current_reg_loss.detach()  
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] =  current_accuracy.detach()  
        
        with open(self.result_path , 'wb+') as f:
             pickle.dump(self.results_dict, f)     
       
        torch.save(net.state_dict(), self.model_path ) 
        
        self.current_running_avg_step += 1
        
        
    def save_test(self, current_accuracy, current_reg_loss, epoch):
        
        self.results_dict["test_accuracy_per_epoch"][epoch] = current_accuracy.detach() 
        self.results_dict["test_loss_per_epoch"][epoch] = current_reg_loss.detach()
        
        with open(self.result_path , 'wb+') as f:
             pickle.dump(self.results_dict, f)
      
        


        
