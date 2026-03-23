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
    
    def __init__(self,  data_manager_obj, train_batch_size, val_batch_size, test_batch_size, eval_frequency):
        
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.eval_frequency = eval_frequency
    
    def evaluvate_network(self, train_context, data_manager_obj, checkpoint_obj, eval_type):
        
         train_context.net.eval()
         
         with torch.no_grad():
             
             task_id = data_manager_obj.current_task_id - data_manager_obj.task_lag 
             
             if eval_type =="task validation":    
                 batch_x, batch_y = data_manager_obj.task_val_x[ task_id ], data_manager_obj.task_mapped_val_y[ task_id ]
                 accuracy_type = "Task Validation Accuracy"
             
             if eval_type =="task test":    
                 batch_x, batch_y = data_manager_obj.task_test_x[ task_id ], data_manager_obj.task_mapped_test_y[ task_id ]
                 accuracy_type = "Task Test Accuracy"
             
             if eval_type =="global validation":    
                 batch_x, batch_y = data_manager_obj.task_val_x[ task_id ], data_manager_obj.task_unmapped_val_y[ task_id ]
                 accuracy_type = "Global Validation Accuracy"
                 
             if eval_type =="global test":    
                  batch_x, batch_y = data_manager_obj.task_test_x[ task_id ], data_manager_obj.task_unmapped_test_y[ task_id ]
                  accuracy_type = "GLOBAL Test Accuracy: "
                  
                  
             predictions = train_context.net.forward(batch_x, data_manager_obj = data_manager_obj, batch_y = batch_y, task_id = task_id, eval_type = eval_type).to(train_context.device) 
            
             accuracy = torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item() *100
                             
         print("Task id ", task_id, accuracy_type , accuracy )
         
         return task_id, accuracy
    
    
    def evaluvate_network_2(self, train_context, data_manager_obj, checkpoint_obj, eval_type):
         
         train_context.net.eval()
         avg_acc = 0.0
         
         with torch.no_grad():
             
           task_id = data_manager_obj.current_task_id - data_manager_obj.task_lag   
         
           for task_id in range( data_manager_obj.current_task_id - data_manager_obj.task_lag , data_manager_obj.current_task_id ):
                
                
                if eval_type =="task validation":    
                    batch_x, batch_y = data_manager_obj.task_val_x[ task_id ], data_manager_obj.task_mapped_val_y[ task_id ]
                    accuracy_type = "Task Validation Accuracy"
                
                if eval_type =="task test":    
                    batch_x, batch_y = data_manager_obj.task_test_x[ task_id ], data_manager_obj.task_mapped_test_y[ task_id ]
                    accuracy_type = "Task Test Accuracy"
                
                
                if eval_type =="global validation":    
                    batch_x, batch_y = data_manager_obj.task_val_x[ task_id ], data_manager_obj.task_unmapped_val_y[ task_id ]
                    accuracy_type = "Global Validation Accuracy"
                    
                if eval_type =="global test":    
                     batch_x, batch_y = data_manager_obj.task_test_x[ task_id ], data_manager_obj.task_unmapped_test_y[ task_id ]
                     accuracy_type = "GLOBAL Test Accuracy: "
                     
                
                predictions = train_context.net.forward(batch_x, data_manager_obj = data_manager_obj, batch_y = batch_y, task_id = task_id, eval_type = eval_type).to(train_context.device) 
               
                accuracy =  torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item() * 100 
                
                avg_acc += accuracy
               
                
         avg_acc = (avg_acc / ( data_manager_obj.task_lag  ))
         
         print("Task id ", data_manager_obj.current_task_id, accuracy_type, avg_acc )   
        
         return data_manager_obj.current_task_id, avg_acc
         

        
    def train(self, train_context, data_manager_obj, checkpoint_obj):

        """train model """
    
        train_context.net.train()
        
        train_accuracy, train_loss = [], []
            
        train_x = data_manager_obj.task_train_x[data_manager_obj.current_task_id]
        
        train_y = data_manager_obj.task_mapped_train_y[data_manager_obj.current_task_id]
        
        for i in range(0, train_x.shape[0], self.train_batch_size ):
            
            batch_x, batch_y = train_x[ i : i + self.train_batch_size], train_y[ i : i + self.train_batch_size]
            
            batch_x = data_manager_obj.augment_batch(batch_x)
            
            for param in train_context.net.parameters(): 
                param.grad = None   # apparently faster than optim.zero_grad()
            
            predictions = train_context.net.forward( batch_x, data_manager_obj = data_manager_obj, batch_y = batch_y, task_id = data_manager_obj.current_task_id, eval_type = "train")
            
            current_reg_loss = train_context.loss(predictions, batch_y)
            
            current_reg_loss.backward()
            
            train_context.optim.step()
        
            train_accuracy.append( torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item())
            
            train_loss.append( current_reg_loss.item())
            
        print("task id ", data_manager_obj.current_task_id, "Train accuracy: ", 100* np.mean(train_accuracy), "Train Loss: ", np.mean(train_loss) )
                
        return np.mean(train_loss)
            
    
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        current_loss = checkpoint_obj.load_experiment_checkpoint(train_context, data_manager_obj)
        
        while data_manager_obj.current_task_id < 50000: 
            
            start = time.perf_counter()
        
            data_manager_obj.create_task_data()
            
            train_loss = self.train( train_context, data_manager_obj, checkpoint_obj)
            
            if train_loss < current_loss:
                
                current_loss = train_loss
                
                checkpoint_obj.save_model_checkpoint(train_context, data_manager_obj, train_loss, data_manager_obj.current_task_id)
                
                
            if ( data_manager_obj.current_task_id >= data_manager_obj.task_lag ) and  ( data_manager_obj.current_task_id % self.eval_frequency == 0 ): 
        
                task_id, task_val_acc = self.evaluvate_network_2(train_context, data_manager_obj, checkpoint_obj, eval_type = "task validation") 
                
                task_id, task_test_acc = self.evaluvate_network_2(train_context, data_manager_obj, checkpoint_obj, eval_type = "task test") 
                
                task_id, global_val_acc = self.evaluvate_network_2(train_context, data_manager_obj, checkpoint_obj, eval_type = "global validation") 
                    
                task_id, global_test_acc = self.evaluvate_network_2(train_context, data_manager_obj, checkpoint_obj, eval_type = "global test") 
                
                checkpoint_obj.save_result_checkpoint(data_manager_obj, train_loss, task_val_acc, task_test_acc, global_val_acc, global_test_acc)
                

            if ( data_manager_obj.current_task_id >= data_manager_obj.task_lag ):
                
                data_manager_obj.delete_data()
                
            data_manager_obj.current_task_id += 1
            
            print("Loop time ", time.perf_counter() -  start)
            
            print("===========================================================================================")
                       
            

       
    
    