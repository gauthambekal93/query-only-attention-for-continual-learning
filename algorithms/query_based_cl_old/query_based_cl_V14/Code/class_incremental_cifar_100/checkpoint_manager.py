# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:13:04 2025

@author: gauthambekal93
"""

import os
import pickle
import torch
import numpy as np
import copy

class CheckpointManager:
    
    def __init__(self, data_manager_obj, runner_obj, root, running_avg_window, model_dir ):
        
        self.create_result_path(root, model_dir)
        
        self.current_running_avg_step, self.running_loss, self.running_accuracy, self.running_avg_window = (0, 0.0, 0.0, running_avg_window)
        
        self.total_updates = np.sum([(i + 1) * data_manager_obj.num_images_per_task * runner_obj.epochs_per_task for i in range(data_manager_obj.total_tasks)]) // runner_obj.train_batch_size
    
        self.results_dict = {}
        self.results_dict["train_loss"] = {}
        self.results_dict["train_accuracy"] = {} 
        self.results_dict["global_test_acc"] = {}
  
        
    def create_result_path(self, root, model_dir):
        os.makedirs( os.path.join(root, model_dir) , exist_ok=True)
        self.result_path = os.path.join(root, model_dir, "result.pkl" )
        self.model_path = os.path.join(root, model_dir ,  "model.pkl")

        
    def save_result_checkpoint(self,  data_manager_obj, train_loss, train_accuracy, global_test_acc):
        
        try:
            current_task_id = data_manager_obj.current_task_id
            
            self.results_dict["task_id"]= current_task_id
            self.results_dict["train_loss"][current_task_id] = train_loss
            
            if "train_accuracy" not in self.results_dict:
                self.results_dict["train_accuracy"] = {}
            
            self.results_dict["train_accuracy"][current_task_id] = train_accuracy
            
            self.results_dict["global_test_acc"][current_task_id] = global_test_acc
            
            self.results_dict["task_train_x"] = copy.deepcopy ( data_manager_obj.task_train_x )
            self.results_dict["task_train_y"] = copy.deepcopy ( data_manager_obj.task_train_y)
            
            self.results_dict["task_val_x"] = copy.deepcopy ( data_manager_obj.task_val_x )
            self.results_dict["task_val_y"] = copy.deepcopy ( data_manager_obj.task_val_y)

            self.results_dict["task_test_x"] = copy.deepcopy ( data_manager_obj.task_test_x)
            self.results_dict["task_test_y"] = copy.deepcopy ( data_manager_obj.task_test_y)

            self.results_dict["buffer"] = copy.deepcopy(data_manager_obj.buffer)
            
          
            with open(self.result_path , 'wb+') as f:
                 pickle.dump(self.results_dict, f) 
        except:
             print("stop")
             print("stop")
             
             
    def save_model_checkpoint(self, train_context, data_manager_obj, train_loss, current_task_id):
        
        checkpoint = {
         "current_task_id":current_task_id,
         "loss": train_loss,
         "model_state": train_context.net.state_dict(),
         "optimizer_state": train_context.optim.state_dict(),
          }         
        
        
        torch.save(checkpoint, self.model_path ) 
        
        
    def load_experiment_checkpoint(self, train_context, data_manager_obj):
        
        if os.path.exists(self.model_path):
            
            checkpoint = torch.load(self.model_path,  map_location = train_context.device)
            
            train_context.net.load_state_dict(checkpoint["model_state"])
            
            train_context.optim.load_state_dict(checkpoint["optimizer_state"])
        
        
        if os.path.exists(self.result_path):        
            
            with open(self.result_path, "rb") as f:
                self.results_dict = pickle.load(f)
            
            data_manager_obj.task_train_x  = self.results_dict["task_train_x"]
            
            data_manager_obj.task_train_y  = self.results_dict["task_train_y"]
            
            
            data_manager_obj.task_val_x  = self.results_dict["task_val_x"]
            
            data_manager_obj.task_val_y  = self.results_dict["task_val_y"]
            
            
            data_manager_obj.task_test_x  = self.results_dict["task_test_x"]
            
            data_manager_obj.task_test_y  = self.results_dict["task_test_y"]
            

            data_manager_obj.buffer =  self.results_dict["buffer"]
            
            task_id =  self.results_dict["task_id"]
            
            data_manager_obj.current_task_id =  task_id + 1
            
            return self.results_dict["train_loss"][task_id]
        
 