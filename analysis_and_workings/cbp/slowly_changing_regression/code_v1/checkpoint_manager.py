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
        self.results_dict["prequential_loss"] = {}
        self.results_dict["forward_loss"] = {}
        self.results_dict["backward_loss"] = {} 
        self.results_dict["effective_rank"] = {} 
    
        
    def create_result_path(self, root, model_dir):
        os.makedirs( os.path.join(root, model_dir) , exist_ok=True)
        self.result_path = os.path.join(root, model_dir, "result.pkl" )
        self.model_path = os.path.join(root, model_dir ,  "model.pkl")

        
    def save_result_checkpoint(self,  data_manager_obj, train_loss, prequential_loss, forward_loss, backward_loss, effective_rank):
        
        try:
            current_task_id = data_manager_obj.current_task_id
            
            self.results_dict["task_id"]= current_task_id
            
            self.results_dict["train_loss"][current_task_id] = train_loss
            
            self.results_dict["prequential_loss"][current_task_id] = prequential_loss
            self.results_dict["forward_loss"][current_task_id] = forward_loss
            self.results_dict["backward_loss"][current_task_id] = backward_loss
            self.results_dict["effective_rank"][current_task_id] = effective_rank
            
            self.results_dict["task_train_x"] = copy.deepcopy ( data_manager_obj.task_train_x )
            self.results_dict["task_train_y"] = copy.deepcopy ( data_manager_obj.task_train_y)
            
            self.results_dict["task_test_x"] = copy.deepcopy ( data_manager_obj.task_test_x )
            self.results_dict["task_test_y"] = copy.deepcopy ( data_manager_obj.task_test_y)
            
            with open(self.result_path , 'wb+') as f:
                 pickle.dump(self.results_dict, f) 
        except:
             print("stop")
             print("stop")
             
             
    def save_model_checkpoint(self, train_context, data_manager_obj, train_loss, current_task_id):
        
        checkpoint = {
         "current_task_id":current_task_id,
         "loss": train_loss,
         "model_state": train_context.learner.net.state_dict(),
         "optimizer_state": train_context.learner.opt.state_dict(),
          }         
        
        
        torch.save(checkpoint, self.model_path ) 
        
        
    def load_experiment_checkpoint(self, train_context, data_manager_obj):
        
        if os.path.exists(self.model_path):
            
            checkpoint = torch.load(self.model_path,  map_location = train_context.device)
            
            train_context.learner.net.load_state_dict(checkpoint["model_state"])
            
            train_context.learner.opt.load_state_dict(checkpoint["optimizer_state"])
        
        
        if os.path.exists(self.result_path):        
            
            with open(self.result_path, "rb") as f:
                self.results_dict = pickle.load(f)
            
            data_manager_obj.task_train_x  = self.results_dict["task_train_x"]
            
            data_manager_obj.task_train_y  = self.results_dict["task_train_y"]
            
            
            data_manager_obj.task_test_x  = self.results_dict["task_test_x"]
            
            data_manager_obj.task_test_y  = self.results_dict["task_test_y"]
            
            
            data_manager_obj.train_loss =  self.results_dict["train_loss"]
            
            data_manager_obj.prequential_loss =  self.results_dict["prequential_loss"]
            
            data_manager_obj.forward_loss =  self.results_dict["forward_loss"]
            
            data_manager_obj.backward_loss =  self.results_dict["backward_loss"]
            
            data_manager_obj.effective_rank =  self.results_dict["effective_rank"]
            
            
            task_id =  self.results_dict["task_id"]
            
            data_manager_obj.current_task_id =  task_id + 1
            
            
            return self.results_dict["train_loss"][task_id]
        
 