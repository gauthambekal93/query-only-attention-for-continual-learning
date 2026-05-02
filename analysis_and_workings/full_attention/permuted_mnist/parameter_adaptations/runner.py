# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:15:04 2025

@author: gauthambekal93
"""


import torch
from tqdm import tqdm
import time
import torch.nn.functional as F
import numpy as np

class Runner:
    
    def __init__(self, num_datapoints_per_timestep, test_batch_size):
        
        self.num_datapoints_per_timestep = num_datapoints_per_timestep
        self.test_batch_size = test_batch_size
        
    
    def prequential_testing(self, train_context, data_manager_obj, batch_x, batch_y):
        
        train_context.net.eval()
        
        with torch.no_grad():
            predictions = train_context.net.prediction(data_manager_obj, batch_x, batch_y)
        
        accuracy = 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
        
        return accuracy
                
    
    def forward_testing(self, train_context, data_manager_obj):
        
        train_context.net.eval()
        
        batch_x, batch_y = data_manager_obj.task_test_x[data_manager_obj.current_task_id], data_manager_obj.task_test_y[ data_manager_obj.current_task_id ]
        
        with torch.no_grad():
            predictions = train_context.net.prediction(data_manager_obj, batch_x, batch_y)
        
        accuracy = 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
        
        return accuracy
        
     
 
    def backward_testing(self, train_context, data_manager_obj):
         
         return 0
     

    def flatten_params(self, train_context):
       return torch.cat([
           p.detach().flatten()
           for p in train_context.net.parameters()
           if p.requires_grad
       ])


    
    def get_param_changes(self, train_context, data_manager_obj):
        
        current_params = self.flatten_params(train_context)
        
        if data_manager_obj.current_task_id > data_manager_obj.num_old_task_window :
            delta = current_params - self.previous_params
            
            param_change  = (torch.norm(delta, p=2) / (delta.numel() ** 0.5)).item()  #torch.norm(current_params - self.previous_params, p=2).item()
        else:
            param_change = 0
        self.previous_params = current_params.clone()
        
        #param_size = torch.norm( current_params,  p=2).item()
        
        d = current_params.numel()

        param_size = (torch.norm(current_params, p=2) / (d ** 0.5)).item()
        
        return param_change, param_size
        
   
    def train(self, train_context, data_manager_obj, checkpoint_obj):

        """train model """
    
        train_context.net.train()
        
        train_accuracy, train_loss, prequential_accuracy  = [], [], []
            
        train_x = data_manager_obj.task_train_x[data_manager_obj.current_task_id]
        
        train_y = data_manager_obj.task_train_y[data_manager_obj.current_task_id]
        
        for i in range(0, train_x.shape[0], self.num_datapoints_per_timestep ):
            
            batch_x  = train_x[ i : i + self.num_datapoints_per_timestep] 
            
            batch_y = train_y[ i : i + self.num_datapoints_per_timestep]
            
            if data_manager_obj.fifo_counter > 0 :
                    
                acc = self.prequential_testing(train_context, data_manager_obj, batch_x, batch_y)
                
                prequential_accuracy.append( acc  )
                
                train_context.net.train()
                
                for param in train_context.net.parameters(): 
                    param.grad = None   # apparently faster than optim.zero_grad()
                
                predictions = train_context.net.prediction( data_manager_obj, batch_x , batch_y)
                   
                current_reg_loss = train_context.loss(predictions, batch_y )
                
                current_reg_loss.backward()
                
                train_context.opt.step()
            
                train_accuracy.append( 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)) )
                
                train_loss.append( current_reg_loss)
            
            data_manager_obj.fill_fifo_buffer( batch_x, batch_y )
    
        train_loss= torch.stack(train_loss).mean().item()
        
        train_accuracy= torch.stack(train_accuracy).mean().item()

        prequential_accuracy = np.mean(prequential_accuracy)
        
        forward_accuracy= self.forward_testing(train_context, data_manager_obj) 
        
        backward_accuracy =  self.backward_testing(train_context, data_manager_obj)
        
        param_change, param_size = self.get_param_changes( train_context, data_manager_obj)
        
        
        print("task id ", data_manager_obj.current_task_id, 
              "Train Loss: ", train_loss,  "Train accuracy: ", train_accuracy,
              "Prequential accuracy", prequential_accuracy, "Forward accuracy: ", forward_accuracy,  "Backward accuracy: ", backward_accuracy,
              "param change", param_change, "param_size", param_size)
        
        return train_loss, train_accuracy, prequential_accuracy, forward_accuracy, backward_accuracy, param_change, param_size
    
        
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        checkpoint_obj.load_experiment_checkpoint(train_context, data_manager_obj)
    
        while data_manager_obj.current_task_id < data_manager_obj.num_tasks: 
            
            start = time.perf_counter()
        
            data_manager_obj.create_task_data()

            if  ( data_manager_obj.current_task_id >= data_manager_obj.num_old_task_window ) :
                
                train_loss, train_accuracy, prequential_accuracy, forward_accuracy, backward_accuracy, param_change, param_size = self.train( train_context, data_manager_obj, checkpoint_obj)
                
                checkpoint_obj.save_model_checkpoint( train_context, data_manager_obj, train_loss, data_manager_obj.current_task_id)
                
                checkpoint_obj.save_result_checkpoint(data_manager_obj, train_loss, train_accuracy, prequential_accuracy, forward_accuracy, backward_accuracy, param_change, param_size)
                
            if data_manager_obj.current_task_id >= data_manager_obj.num_old_task_window: 
                
                data_manager_obj.delete_data()
                
                 
            data_manager_obj.current_task_id += 1
            
            print("Loop time ", time.perf_counter() -  start)
            
            print("===========================================================================================")
            
           

