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
    
    def __init__(self, data_manager_obj, runner_obj, root, model_dir ):
        
        self.create_result_path(root, model_dir)

        self.results_dict = {}
        self.results_dict["train_loss"] = {}
        self.results_dict["task_val_acc"] = {}
        self.results_dict["task_test_acc"] = {}
        self.results_dict["global_val_acc"] = {}
        self.results_dict["global_test_acc"] = {}
   
        
    def create_result_path(self, root, model_dir):
        os.makedirs( os.path.join(root, model_dir) , exist_ok=True)
        self.result_path = os.path.join(root, model_dir, "result.pkl" )
        self.model_path = os.path.join(root, model_dir ,  "model.pkl")
    
    '''
    def save_result_checkpoint(self,  data_manager_obj, train_loss, task_val_acc, task_test_acc, global_val_acc, global_test_acc):
        
        try:
            current_task_id = data_manager_obj.current_task_id
            
            self.results_dict["task_id"]= current_task_id
            self.results_dict["train_loss"][current_task_id] = train_loss
            self.results_dict["task_val_acc"][current_task_id] = task_val_acc
            self.results_dict["task_test_acc"][current_task_id] = task_test_acc
            self.results_dict["global_val_acc"][current_task_id] = global_val_acc
            self.results_dict["global_test_acc"][current_task_id] = global_test_acc
            
            
            self.results_dict["task_train_x"] = copy.deepcopy ( data_manager_obj.task_train_x )
            self.results_dict["task_unmapped_train_y"] = copy.deepcopy ( data_manager_obj.task_unmapped_train_y)
            self.results_dict["task_mapped_train_y"] = copy.deepcopy ( data_manager_obj.task_mapped_train_y)
            self.results_dict["task_val_x"] = copy.deepcopy ( data_manager_obj.task_val_x )
            self.results_dict["task_unmapped_val_y"] = copy.deepcopy ( data_manager_obj.task_unmapped_val_y)
            self.results_dict["task_mapped_val_y"] = copy.deepcopy ( data_manager_obj.task_mapped_val_y)
            self.results_dict["task_test_x"] = copy.deepcopy ( data_manager_obj.task_test_x)
            self.results_dict["task_unmapped_test_y"] = copy.deepcopy ( data_manager_obj.task_unmapped_test_y)
            self.results_dict["task_mapped_test_y"] = copy.deepcopy ( data_manager_obj.task_mapped_test_y)
            self.results_dict["local_eval_support_x"] = copy.deepcopy ( data_manager_obj.local_eval_support_x)
            self.results_dict["local_eval_support_y"] = copy.deepcopy ( data_manager_obj.local_eval_support_y)
            self.results_dict["global_data"] = copy.deepcopy(data_manager_obj.global_data)
            
          
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
        
    '''
    
    def load_experiment_checkpoint(self, train_context, data_manager_obj):
        
        if os.path.exists(self.model_path):
            
            checkpoint = torch.load(self.model_path,  map_location = train_context.device)
            
            train_context.net.load_state_dict(checkpoint["model_state"])
            
            train_context.optim.load_state_dict(checkpoint["optimizer_state"])
        
        
        if os.path.exists(self.result_path):        
            
            with open(self.result_path, "rb") as f:
                self.results_dict = pickle.load(f)
            
            data_manager_obj.task_train_x  = self.results_dict["task_train_x"]
            
            data_manager_obj.task_unmapped_train_y  = self.results_dict["task_unmapped_train_y"]
            
            data_manager_obj.task_mapped_train_y  = self.results_dict["task_mapped_train_y"]
            
            data_manager_obj.task_val_x  = self.results_dict["task_val_x"]
            
            data_manager_obj.task_unmapped_val_y  = self.results_dict["task_unmapped_val_y"]
            
            data_manager_obj.task_mapped_val_y  = self.results_dict["task_mapped_val_y"]
            
            data_manager_obj.task_test_x  = self.results_dict["task_test_x"]
            
            data_manager_obj.task_unmapped_test_y  = self.results_dict["task_unmapped_test_y"]
            
            data_manager_obj.task_mapped_test_y  = self.results_dict["task_mapped_test_y"]
            
            data_manager_obj.local_eval_support_x  = self.results_dict["local_eval_support_x"]
            
            data_manager_obj.local_eval_support_y  = self.results_dict["local_eval_support_y"]
            
            data_manager_obj.global_data =  self.results_dict["global_data"]
            
            task_id =  self.results_dict["task_id"]
            
            data_manager_obj.current_task_id =  task_id + 1
            
            return self.results_dict["train_loss"][task_id]
        
        else:
            
            return np.inf
        
    
        
