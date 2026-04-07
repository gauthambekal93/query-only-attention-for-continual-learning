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
        self.results_dict["train_accuracy"] = {}
        self.results_dict["prequential_accuracy"] = {}
        self.results_dict["forward_accuracy"] = {}
        self.results_dict["backward_accuracy"] = {} 
        
    
        
    def create_result_path(self, root, model_dir):
        os.makedirs( os.path.join(root, model_dir) , exist_ok=True)
        self.result_path = os.path.join(root, model_dir, "result.pkl" )
        self.model_path = os.path.join(root, model_dir ,  "model.pkl")

        
    def save_result_checkpoint(self,  data_manager_obj, train_loss, train_accuracy, prequential_accuracy, forward_accuracy, backward_accuracy):
        
        try:
            current_task_id = data_manager_obj.current_task_id
            
            self.results_dict["task_id"]= current_task_id
            
            self.results_dict["train_loss"][current_task_id] = train_loss
            self.results_dict["train_accuracy"][current_task_id] = train_accuracy
            
            self.results_dict["prequential_accuracy"][current_task_id] = prequential_accuracy
            self.results_dict["forward_accuracy"][current_task_id] = forward_accuracy
            self.results_dict["backward_accuracy"][current_task_id] = backward_accuracy
            
            self.results_dict["task_train_x"] = copy.deepcopy ( data_manager_obj.task_train_x )
            self.results_dict["task_train_y"] = copy.deepcopy ( data_manager_obj.task_train_y)
            
            self.results_dict["task_test_x"] = copy.deepcopy ( data_manager_obj.task_test_x )
            self.results_dict["task_test_y"] = copy.deepcopy ( data_manager_obj.task_test_y)

            self.results_dict["fifo_x"] = copy.deepcopy(data_manager_obj.fifo_x)
            self.results_dict["fifo_y"] = copy.deepcopy(data_manager_obj.fifo_y)
            
            self.results_dict["fifo_counter"] = data_manager_obj.fifo_counter
            
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
         "optimizer_state": train_context.opt.state_dict(),
          }         
        
        
        torch.save(checkpoint, self.model_path ) 
        
        
    def load_experiment_checkpoint(self, train_context, data_manager_obj):
        
        if os.path.exists(self.model_path):
            
            checkpoint = torch.load(self.model_path,  map_location = train_context.device)
            
            train_context.net.load_state_dict(checkpoint["model_state"])
            
            train_context.opt.load_state_dict(checkpoint["optimizer_state"])
        
        
        if os.path.exists(self.result_path):        
            
            with open(self.result_path, "rb") as f:
                self.results_dict = pickle.load(f)
            
            data_manager_obj.task_train_x  = self.results_dict["task_train_x"]
            
            data_manager_obj.task_train_y  = self.results_dict["task_train_y"]
            
            
            data_manager_obj.task_test_x  = self.results_dict["task_test_x"]
            
            data_manager_obj.task_test_y  = self.results_dict["task_test_y"]
        
            
            
            data_manager_obj.fifo_x =  self.results_dict["fifo_x"]
            
            data_manager_obj.fifo_y =  self.results_dict["fifo_y"]
            
            
            data_manager_obj.buffer_counter = self.results_dict["buffer_counter"]
            
            data_manager_obj.fifo_counter = self.results_dict["fifo_counter"] 
            
            data_manager_obj.train_loss =  self.results_dict["train_loss"]
            
            data_manager_obj.train_accuracy =  self.results_dict["train_accuracy"]
            
            data_manager_obj.prequential_accuracy =  self.results_dict["prequential_accuracy"]
            
            data_manager_obj.forward_accuracy =  self.results_dict["forward_accuracy"]
            
            data_manager_obj.backward_accuracy =  self.results_dict["backward_accuracy"]
            
            
            task_id =  self.results_dict["task_id"]
            
            data_manager_obj.current_task_id =  task_id + 1
            
            
            return self.results_dict["train_loss"][task_id]
'''        
for i in self.results_dict["task_test_x"].keys():
    print(self.results_dict["task_test_x"][i].shape, self.results_dict["task_test_x"][i].dtype)
    
    
    
    
for i in self.results_dict["buffer_x"].keys():
    x = self.results_dict["buffer_x"][i]
    print(
        i,
        "shape =", x.shape,
        "visible_bytes =", x.numel() * x.element_size(),
        "storage_bytes =", x.untyped_storage().nbytes(),
        "is_view =", x._base is not None,
        "is_contiguous =", x.is_contiguous()
    )    
    
'''    
    
    
    
    
    
    
    