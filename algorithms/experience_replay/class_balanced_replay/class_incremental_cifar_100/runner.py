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
    
    def __init__(self,  data_manager_obj, num_epochs, train_batch_size, val_batch_size, test_batch_size):
        
        self.num_epochs = num_epochs
        self.epochs_per_task = int( self.num_epochs / data_manager_obj.total_tasks ) #epochs_per_task
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
      
                    
    
    def test_network(self, train_context, data_manager_obj, checkpoint_obj):
         
         train_context.net.eval()
         
         avg_acc = 0.0
         sub_task_accuracies = {}

         with torch.no_grad():
             
             for task_id in data_manager_obj.task_test_x.keys():
                 
                 batch_x, batch_y = data_manager_obj.task_test_x[ task_id ], data_manager_obj.task_test_y[ task_id ]
                  
                 predictions = train_context.net.forward( X = batch_x, Y = batch_y, data_manager_obj = data_manager_obj)
                 
                 accuracy = torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
                 
                 avg_acc += accuracy
                 
                 sub_task_accuracies[task_id] = accuracy

                 
         accuracy =  100 * (avg_acc / ( len(data_manager_obj.task_test_x.keys() ) ) )
          
         print("Test task accuracy: ", accuracy  )
         
         return data_manager_obj.current_task_id, accuracy
         
            
    def train(self, train_context, data_manager_obj, checkpoint_obj):

        """train model """
    
        train_context.net.train()
        
        train_accuracy, train_loss = [], []
            
        train_x = data_manager_obj.task_train_x[data_manager_obj.current_task_id]
        
        train_y = data_manager_obj.task_train_y[data_manager_obj.current_task_id]
        
        #train_labels 
        for i in range(0, train_x.shape[0], self.train_batch_size ):
            
            batch_x, batch_y = train_x[ i : i + self.train_batch_size], train_y[ i : i + self.train_batch_size] 
            
            data_manager_obj.fill_buffer(  batch_x , batch_y)
            
            if sum(data_manager_obj.is_filled) < 100:
                
                continue
            
            data_manager_obj.create_unique_labels()
                
            replay_x, replay_y =  data_manager_obj.get_data()
            
            X, Y = torch.cat([batch_x, replay_x]), torch.cat([batch_y,replay_y])
            
            rand_ids = torch.randperm(len(X))
            
            X, Y = X[rand_ids], Y[rand_ids]
            
            for param in train_context.net.parameters(): 
                param.grad = None   # apparently faster than optim.zero_grad()
            
            predictions = train_context.net.forward( X = X, Y = Y, data_manager_obj = data_manager_obj)
               
            current_reg_loss = train_context.loss(predictions, Y )
            
            current_reg_loss.backward()
            
            train_context.optim.step()
        
            train_accuracy.append( torch.mean((predictions.argmax(axis=1) == Y).to(torch.float32)) )
            
            train_loss.append( current_reg_loss)
        
        
        if sum(data_manager_obj.is_filled) ==100:    
            
            train_loss = torch.mean( torch.stack( train_loss )).item() 
        
            train_accuracy = 100*  torch.mean( torch.stack(train_accuracy) ).item() 
            
            print("task id ", data_manager_obj.current_task_id, "Train accuracy: ", train_accuracy, "Train Loss: ", train_loss )
            return train_loss , train_accuracy
        else:
             return None, None 
            
            
    
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        checkpoint_obj.load_experiment_checkpoint(train_context, data_manager_obj)
    
        while data_manager_obj.current_task_id < 50000: #data_manager_obj.total_tasks:
            
            start = time.perf_counter()
        
            data_manager_obj.create_task_data()
            
            train_loss , train_accuracy = self.train( train_context, data_manager_obj, checkpoint_obj)
            
            
            if ( sum(data_manager_obj.is_filled) == 100 ) and ( data_manager_obj.current_task_id % 20 == 0 ):
                
                task_id, global_test_acc = self.test_network( train_context, data_manager_obj, checkpoint_obj)
                
                checkpoint_obj.save_model_checkpoint( train_context, data_manager_obj, train_loss, data_manager_obj.current_task_id)
                
                checkpoint_obj.save_result_checkpoint(data_manager_obj, train_loss, train_accuracy, global_test_acc)
                
            if data_manager_obj.current_task_id >=20: 
                
                data_manager_obj.delete_data()
                
                 
            data_manager_obj.current_task_id += 1
            
            print("Loop time ", time.perf_counter() -  start)
            
            print("===========================================================================================")
            
           

