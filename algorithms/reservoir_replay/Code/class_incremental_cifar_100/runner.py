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
    
    def __init__(self,  data_manager_obj, num_epochs, incoming_batch_size, mixing_ratio, val_batch_size, test_batch_size):
        
        self.num_epochs = num_epochs
        self.incoming_batch_size = incoming_batch_size
        self.mixing_ratio = mixing_ratio
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
      
    
    def set_lr(self, epoch, train_context):
            """ Changes the learning rate of the optimizer according to the current epoch of the task """
            current_stepsize = None
            
            if epoch  == 0:
                current_stepsize = train_context.step_size

            elif epoch == 200:
                current_stepsize = round(train_context.step_size * 0.2, 5)
            
            elif epoch == 800:
                current_stepsize = round(train_context.step_size * (0.2 ** 2), 5)
            
            elif epoch == 1600:
                current_stepsize = round(train_context.step_size * (0.2 ** 3), 5)

            if current_stepsize is not None:
                for g in train_context.optim.param_groups:
                    g['lr'] = current_stepsize
                    
    
    def eval_network(self, train_context, data_manager_obj, checkpoint_obj, eval_type):
         
         train_context.net.eval()
         
         avg_acc = 0.0
         sub_task_accuracies = {}

         with torch.no_grad():
             
             for task_id in data_manager_obj.task_test_x.keys():
                 
                 if eval_type =="global validation":    
                     batch_x, batch_y = data_manager_obj.task_val_x[ task_id ], data_manager_obj.task_val_y[ task_id ]
                     accuracy_type = "Global Validation Accuracy"
                     
                 if eval_type =="global test":    
                     batch_x, batch_y = data_manager_obj.task_test_x[ task_id ], data_manager_obj.task_test_y[ task_id ]
                     accuracy_type = "Global Test Accuracy"
                     
                 predictions = train_context.net.forward(batch_x, data_manager_obj = data_manager_obj, batch_y = batch_y, task_id = task_id).to(train_context.device) 
                 
                 accuracy = torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item() * 100
                 
                 avg_acc += accuracy
                 
                 sub_task_accuracies[task_id] = accuracy
    
         avg_acc = (avg_acc / ( data_manager_obj.task_lag  ))
         
         print("Task id ", data_manager_obj.current_task_id, accuracy_type, avg_acc )   
         
         return data_manager_obj.current_task_id, avg_acc
         

        
             
            
    def train(self, train_context, data_manager_obj, checkpoint_obj):

        """train model """
    
        train_context.net.train()
        
        train_accuracy, train_loss = [], []
            
        train_x = data_manager_obj.task_train_x[data_manager_obj.current_task_id]
        
        train_y = data_manager_obj.task_train_y[data_manager_obj.current_task_id]
        
        for i in range(0, train_x.shape[0], self.incoming_batch_size ):
            
            #if data_manager_obj.step >= data_manager_obj.buffer_size - data_manager_obj.incoming_batch_size:
                
            if data_manager_obj.buffer_count == data_manager_obj.buffer_size  :
                
                X, Y = data_manager_obj.sample_buffer()
                
                batch_x = train_x[ i : i + int(self.incoming_batch_size * self.mixing_ratio) ]
                
                batch_y = train_y[ i : i + int(self.incoming_batch_size * self.mixing_ratio) ]
                
                batch_x, batch_y = torch.cat( (batch_x, X) ), torch.cat( (batch_y, Y) )
                
                rand_idx = torch.randperm(len(batch_x))
                
                batch_x, batch_y =  batch_x[rand_idx], batch_y[rand_idx]
                
                for param in train_context.net.parameters(): 
                    param.grad = None   # apparently faster than optim.zero_grad()
                
                predictions = train_context.net.forward( batch_x, data_manager_obj = data_manager_obj, batch_y = batch_y, task_id = data_manager_obj.current_task_id)
                
                current_reg_loss = train_context.loss(predictions, batch_y)
                
                current_reg_loss.backward()
                
                train_context.optim.step()
                
                train_accuracy.append( torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item())
                
                train_loss.append( current_reg_loss.item())
                
            data_manager_obj.fill_buffer( train_x[ i : i + self.incoming_batch_size], train_y[ i : i + self.incoming_batch_size] )
        
        if data_manager_obj.step>=data_manager_obj.buffer_size:
            
            if len(train_loss)==0:
                return np.inf
            else:
                print("task id ", data_manager_obj.current_task_id, "Train accuracy: ", 100* np.mean(train_accuracy), "Train Loss: ", np.mean(train_loss) )
                return np.mean(train_loss)
      
          
        
            
    
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        current_loss = checkpoint_obj.load_experiment_checkpoint(train_context, data_manager_obj)
        
        while data_manager_obj.current_task_id < 50000: #data_manager_obj.total_tasks:
            
            start = time.perf_counter()
        
            data_manager_obj.create_task_data()
            
            train_loss = self.train( train_context, data_manager_obj, checkpoint_obj)
            
            if train_loss < current_loss:
                
                current_loss = train_loss
     
                checkpoint_obj.save_model_checkpoint(train_context, data_manager_obj, train_loss, data_manager_obj.current_task_id)
     
            if ( data_manager_obj.current_task_id >= data_manager_obj.task_lag ):
                
                task_id, global_val_acc = self.eval_network(train_context, data_manager_obj, checkpoint_obj, eval_type ="global validation") 
                    
                task_id, global_test_acc = self.eval_network( train_context, data_manager_obj, checkpoint_obj, eval_type ="global test")
                
                checkpoint_obj.save_result_checkpoint(data_manager_obj, train_loss, global_val_acc, global_test_acc)
                
            if ( data_manager_obj.current_task_id >= data_manager_obj.task_lag ):
                
                data_manager_obj.delete_data()
                
            data_manager_obj.current_task_id += 1
            
            print("Loop time ", time.perf_counter() -  start)
            
            print("===========================================================================================")
            
           

