# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:13:04 2025

@author: gauthambekal93
"""

import os
import pickle
import torch
import numpy as np

class CheckpointManager:
    
    def __init__(self, data_manager_obj, runner_obj, root, running_avg_window, model_dir ):
        
        
        self.result_dir = os.path.join(root, model_dir )
        
        self.current_running_avg_step, self.running_loss, self.running_accuracy, self.running_avg_window = (0, 0.0, 0.0, running_avg_window)
        
        #bin_size = int( (running_avg_window * runner_obj.train_batch_size ))
        
        #num_images_per_task = num_images_per_class * data_manager_obj.class_increase_per_task
        
        total_updates = np.sum([(i + 1) * data_manager_obj.num_images_per_task * runner_obj.epochs_per_task for i in range(data_manager_obj.total_tasks)]) // runner_obj.train_batch_size
        
        self.results_dict = {}
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(  int(total_updates / running_avg_window) )
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros(  int(total_updates / running_avg_window) )
        
        self.results_dict["test_loss_per_epoch"] = torch.zeros( runner_obj.num_epochs )
        self.results_dict["test_accuracy_per_epoch"] =  torch.zeros(  runner_obj.num_epochs )
        
        self.model_dir = os.path.join(root, model_dir )
        
    def create_result_dir(self):
        os.makedirs( self.model_dir , exist_ok=True)
    
    
    def summarize_train(self):
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] =  self.running_loss
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] =  self.running_accuracy
    
        self.current_running_avg_step += 1
        self.running_loss *= 0.0 
        self.running_accuracy *= 0.0
        
        
    def summarize_test(self, current_reg_loss, current_accuracy, epoch):     
        self.results_dict["test_loss_per_epoch"][epoch] = current_reg_loss.detach()
        self.results_dict["test_accuracy_per_epoch"][epoch] = current_accuracy.detach() 

        
    def save_experiment_checkpoint(self, train_context):
        
        with open(self.result_path , 'wb+') as f:
             pickle.dump(self.results_dict, f)     
       
        torch.save(train_context.state_dict(), self.model_path ) 
        
    

