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
    
    def __init__(self, num_datapoints_per_timestep):
        
        self.num_datapoints_per_timestep = num_datapoints_per_timestep
        
    
    def prequential_testing(self, train_context, data_manager_obj, batch_x, batch_y):
        
        train_context.net.eval()
        
        with torch.no_grad():
            predictions = train_context.net.forward_prediction(data_manager_obj, batch_x)
        
        accuracy = 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
        
        return accuracy
                
    
    def forward_testing(self,train_context, data_manager_obj):
        
        train_context.net.eval()
        
        batch_x, batch_y = data_manager_obj.task_test_x[data_manager_obj.current_task_id], data_manager_obj.task_test_y[ data_manager_obj.current_task_id ]
        
        with torch.no_grad():
            predictions = train_context.net.forward_prediction( data_manager_obj, batch_x)
        
        accuracy = 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
        
        return accuracy
        
     
 
    def backward_testing(self, train_context, data_manager_obj):
         
         return 0
         
         train_context.net.eval()
         
         avg_acc = 0.0
         sub_task_accuracies = {}

         with torch.no_grad():
             
             for task_id in data_manager_obj.task_test_x.keys():
                 
                 if task_id != data_manager_obj.current_task_id:
                 
                     batch_x, batch_y = data_manager_obj.task_test_x[ task_id ], data_manager_obj.task_test_y[ task_id ]
                      
                     predictions = train_context.net.backward_prediction( data_manager_obj,  batch_x)
                     
                     accuracy = 100 * torch.mean((predictions == batch_y).to(torch.float32)).item()
                     
                     avg_acc += accuracy
                     
                     sub_task_accuracies[task_id] = accuracy
  
         accuracy =  avg_acc / ( len(data_manager_obj.task_test_x.keys() ) - 1 ) 
         
         return accuracy
         
    ''' 
    def predict_task_boundary(self, data_manager_obj, train_context, X, Y ):
        
        threshold_diff = 30
        
        if data_manager_obj.fifo_y[0].sum()>0:
            acc0 = self.prequential_testing(train_context, data_manager_obj, X, Y, fifo_id = 0)
        else:
            acc0 = 100 * ( 1/ data_manager_obj.classes_per_task)
            
        acc1 = self.prequential_testing(train_context, data_manager_obj, X, Y, fifo_id = 1)
        
        if  ( np.abs( acc1- acc0) > threshold_diff ):
            new_task = True
        else:
            new_task = False
            
        return new_task
    '''
    
        
    def train(self, train_context, data_manager_obj, checkpoint_obj):

        """train model """
    
        train_context.net.train()
        
        train_accuracy, train_loss, prequential_accuracy = [], [], []
            
        train_x = data_manager_obj.task_train_x[data_manager_obj.current_task_id]
        
        train_y = data_manager_obj.task_train_y[data_manager_obj.current_task_id]
        
        for i in range(0, train_x.shape[0], self.num_datapoints_per_timestep ):
 
            batch_x  = train_x[ i : i + self.num_datapoints_per_timestep] 
            
            batch_y = train_y[ i : i + self.num_datapoints_per_timestep]
            
            #data_manager_obj.fill_fifo_buffer( batch_x, batch_y)
            
            if data_manager_obj.fifo_counter > 0: 
                
                #if len(torch.unique(data_manager_obj.fifo_y[1])) < data_manager_obj.classes_per_task:
                #    continue
                
                acc = self.prequential_testing( train_context, data_manager_obj, batch_x, batch_y)
                
                prequential_accuracy.append( acc  )
                
                
                train_context.net.train()
                
                for param in train_context.net.parameters(): 
                    param.grad = None   # apparently faster than optim.zero_grad()
                
                predictions = train_context.net.forward_prediction(data_manager_obj, batch_x )
                   
                current_reg_loss = train_context.loss(predictions, batch_y )
                
                current_reg_loss.backward()
                
                train_context.opt.step()
            
                train_accuracy.append( 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)) )
                
                train_loss.append( current_reg_loss)
               
            data_manager_obj.fill_fifo_buffer( batch_x, batch_y)
               
    
        train_loss= torch.stack(train_loss).mean().item()
        
        train_accuracy= torch.stack(train_accuracy).mean().item()

        prequential_accuracy = np.mean(prequential_accuracy)
        
        forward_accuracy= self.forward_testing( train_context, data_manager_obj) 
        
        backward_accuracy =  self.backward_testing(train_context, data_manager_obj)
        
        print("task id ", data_manager_obj.current_task_id, 
              "Train Loss: ", train_loss,  "Train accuracy: ", train_accuracy,
              "Prequential accuracy", prequential_accuracy, "Forward accuracy: ", forward_accuracy,  "Backward accuracy: ", backward_accuracy )
        
        return train_loss, train_accuracy, prequential_accuracy, forward_accuracy, backward_accuracy
    
        
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        checkpoint_obj.load_experiment_checkpoint(train_context, data_manager_obj)
    
        while data_manager_obj.current_task_id < data_manager_obj.num_tasks: 
            
            start = time.perf_counter()
        
            data_manager_obj.create_task_data()

            if  ( data_manager_obj.current_task_id >= data_manager_obj.num_old_task_window ) :
                
                train_loss, train_accuracy, prequential_accuracy, forward_accuracy, backward_accuracy = self.train( train_context, data_manager_obj, checkpoint_obj)
                
                checkpoint_obj.save_model_checkpoint( train_context, data_manager_obj, train_loss, data_manager_obj.current_task_id)
                
                checkpoint_obj.save_result_checkpoint(data_manager_obj, train_loss, train_accuracy, prequential_accuracy, forward_accuracy, backward_accuracy)
                
            if data_manager_obj.current_task_id >= data_manager_obj.num_old_task_window: 
                
                data_manager_obj.delete_data()
                
                 
            data_manager_obj.current_task_id += 1
            
            print("Loop time ", time.perf_counter() -  start)
            
            print("===========================================================================================")
            
           

