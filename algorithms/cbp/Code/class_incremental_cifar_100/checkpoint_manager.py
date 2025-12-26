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
        
        self.create_result_path(root, model_dir)
        
        self.current_running_avg_step, self.running_loss, self.running_accuracy, self.running_avg_window = (0, 0.0, 0.0, running_avg_window)
        
        self.total_updates = np.sum([(i + 1) * data_manager_obj.num_images_per_task * runner_obj.epochs_per_task for i in range(data_manager_obj.total_tasks)]) // runner_obj.train_batch_size
        
        self.results_dict = {}
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros(  int(self.total_updates / running_avg_window) )
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros(  int(self.total_updates / running_avg_window) )
        
        self.results_dict["test_loss_per_task"] = torch.zeros( data_manager_obj.total_tasks )
        self.results_dict["test_accuracy_per_task"] =  torch.zeros( data_manager_obj.total_tasks )
        
        self.results_dict["current_task_id"]  = data_manager_obj.current_task_id
        self.results_dict["current_running_avg_step"]  = self.current_running_avg_step
        self.results_dict['current_num_classes'] = data_manager_obj.current_num_classes
        
    def create_result_path(self, root, model_dir):
        os.makedirs( os.path.join(root , model_dir ) , exist_ok=True)
        self.result_path = os.path.join(root, model_dir, "result.pkl" )
        self.model_path = os.path.join(root, model_dir ,  "model.pkl")
    
    def summarize_train(self):
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] =  self.running_loss / self.running_avg_window
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] = 100 * (  self.running_accuracy /self.running_avg_window )
    
        self.current_running_avg_step += 1
        self.running_loss *= 0.0 
        self.running_accuracy *= 0.0
        
        
    def summarize_test(self, current_reg_loss, current_accuracy, current_task_id, current_num_classes):     
        self.results_dict["test_loss_per_task"][current_task_id] = current_reg_loss.detach()
        self.results_dict["test_accuracy_per_task"][current_task_id] = current_accuracy.detach() 

        self.results_dict["current_task_id"] = current_task_id
        self.results_dict["current_running_avg_step"] = self.current_running_avg_step
        self.results_dict['current_num_classes'] = current_num_classes
        
    def save_experiment_checkpoint(self, train_context):
        
        with open(self.result_path , 'wb+') as f:
             pickle.dump(self.results_dict, f)     
        
        checkpoint = {
        "model_state": train_context.net.state_dict(),
        "optimizer_state": train_context.optim.state_dict(),
        "resgnt": train_context.resgnt
        }

        torch.save(checkpoint, self.model_path ) 
        
    
    def load_experiment_checkpoint(self, train_context, data_manager_obj):
        
        checkpoint = torch.load(self.model_path,  map_location = train_context.device)
        
        train_context.net.load_state_dict(checkpoint["model_state"])
        
        train_context.optim.load_state_dict(checkpoint["optimizer_state"])
        
        train_context.resgnt = checkpoint["resgnt"]
        
        with open(self.result_path, "rb") as f:
            self.results_dict = pickle.load(f)
    
        """We want to update the current running step and task id where the training was previously stopped and saved. """
        self.current_running_avg_step = self.results_dict['current_running_avg_step']
        data_manager_obj.current_task_id = self.results_dict['current_task_id'] 
        data_manager_obj.current_num_classes = self.results_dict['current_num_classes']     
        
