# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:15:04 2025

@author: gauthambekal93
"""


import torch
import time
import numpy as np

class Runner:
    
    def __init__(self, num_datapoints_per_timestep):
        
        self.num_datapoints_per_timestep = num_datapoints_per_timestep
        
      
                    
    def prequential_testing(self, train_context, batch_x, batch_y):
        
        train_context.net.eval()
        
        with torch.no_grad():
            predictions = train_context.net.forward(x=batch_x)
        
        loss = train_context.loss(predictions, batch_y ).item()
        
        return loss
                

    def forward_testing(self, train_context, data_manager_obj):
          
          train_context.net.eval()
          
          batch_x, batch_y = data_manager_obj.task_test_x[data_manager_obj.current_task_id], data_manager_obj.task_test_y[ data_manager_obj.current_task_id ]
          
          with torch.no_grad():
              predictions = train_context.net.forward(x=batch_x)
          
          loss = train_context.loss(predictions, batch_y ).item()
          
          return loss
      
        
    def backward_testing(self, train_context, data_manager_obj):
         
        return 5
    
         
    def get_effective_rank(self, train_context, data_manager_obj):
         
        batch_x, batch_y = data_manager_obj.task_test_x[data_manager_obj.current_task_id], data_manager_obj.task_test_y[ data_manager_obj.current_task_id ]
        
        effective_rank = train_context.net.calculate_effective_rank(batch_x, batch_y, train_context.loss)
        
        return effective_rank
         
            
    def train(self, train_context, data_manager_obj, checkpoint_obj):

        """train model """
    
        train_context.net.train()
        
        train_loss, prequential_loss = [], []
            
        train_x = data_manager_obj.task_train_x[data_manager_obj.current_task_id]
        
        train_y = data_manager_obj.task_train_y[data_manager_obj.current_task_id]
        
        for i in range(0, train_x.shape[0], self.num_datapoints_per_timestep ):
            
            batch_x, batch_y = train_x[ i : i + self.num_datapoints_per_timestep], train_y[ i : i + self.num_datapoints_per_timestep] 
            
            prequential_loss.append(  self.prequential_testing(train_context, batch_x, batch_y) )
            
            train_context.net.train()
            
            for param in train_context.net.parameters(): 
                param.grad = None   # apparently faster than optim.zero_grad()
            
            predictions = train_context.net.forward( x = batch_x)
               
            current_reg_loss = train_context.loss(predictions, batch_y )
            
            current_reg_loss.backward()
            
            train_context.opt.step()
            
            train_loss.append( current_reg_loss)
        
        train_loss= torch.stack(train_loss).mean().item()

        prequential_loss = np.mean(prequential_loss)
        
        forward_loss =  self.forward_testing(train_context, data_manager_obj)
        
        backward_loss =  self.backward_testing(train_context, data_manager_obj)
        
        effective_rank =  self.get_effective_rank(train_context, data_manager_obj)
        
        print("task id ", data_manager_obj.current_task_id, 
              "Train Loss: ", train_loss,  "Prequential loss", prequential_loss, "Forward loss", forward_loss,  "Backward loss: ", backward_loss,
              "Effective Rank: ", effective_rank)
        
        return train_loss, prequential_loss, forward_loss, backward_loss, effective_rank
            
            
    
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        checkpoint_obj.load_experiment_checkpoint(train_context, data_manager_obj)
    
        while data_manager_obj.current_task_id < data_manager_obj.num_tasks:
            
            start = time.perf_counter()
        
            data_manager_obj.create_task_data()
            
            if  ( data_manager_obj.current_task_id >= data_manager_obj.num_old_task_window ) :
                
                train_loss, prequential_loss, forward_loss, backward_loss,effective_rank = self.train( train_context, data_manager_obj, checkpoint_obj)
                
                checkpoint_obj.save_model_checkpoint( train_context, data_manager_obj, train_loss, data_manager_obj.current_task_id)
                
                checkpoint_obj.save_result_checkpoint(data_manager_obj, train_loss, prequential_loss, forward_loss, backward_loss, effective_rank)
                
            if data_manager_obj.current_task_id >= data_manager_obj.num_old_task_window: 
                
                data_manager_obj.delete_data()
                
                 
            data_manager_obj.current_task_id += 1
            
            print("Loop time ", time.perf_counter() -  start)
            
            print("===========================================================================================")
            
           

