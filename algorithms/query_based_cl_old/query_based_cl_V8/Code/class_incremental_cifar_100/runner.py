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
                    

    def test_network(self, train_context, data_manager_obj, checkpoint_obj):
         
         train_context.net.eval()
         
         avg_acc = 0.0
         sub_task_accuracies = {}

         with torch.no_grad():
             
             for task_id in data_manager_obj.cifar_test_x.keys():
                 
                 if task_id != 5:
                     continue
                 
                 batch_x, batch_y = data_manager_obj.cifar_test_x[ task_id ], data_manager_obj.cifar_test_y[ task_id ]
                 
                 predictions = train_context.net.forward_test(batch_x, data_manager_obj = data_manager_obj, batch_y = batch_y, task_id = task_id).to(train_context.device) 
                 
                 accuracy = torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
                 
                 avg_acc += accuracy
                 
                 sub_task_accuracies[task_id] = accuracy
               
         #checkpoint_obj.summarize_test( avg_acc / (data_manager_obj.current_task_id + 1 ) , sub_task_accuracies, data_manager_obj.current_task_id, data_manager_obj.current_num_classes )
         
         print("Test task accuracy: ", 100 * (avg_acc / (data_manager_obj.current_task_id + 1 ) )  )
         
         print("Test sub-task accuracy: ", { task_id: 100*accuracy for task_id, accuracy in sub_task_accuracies.items() }  )
     
                     
    def val_network(self, train_context, data_manager_obj, checkpoint_obj):
         
         train_context.net.eval()

         #avg_acc = 0.0
         #sub_task_accuracies = {}
         
         with torch.no_grad():
             task_id = data_manager_obj.current_task_id - 20
             
             #for task_id in data_manager_obj.task_val_x[data_manager_obj.current_task_id-5]: # data_manager_obj.task_val_x.keys():
            
             batch_x, batch_y = data_manager_obj.task_val_x[ task_id ], data_manager_obj.task_val_y[ task_id ]
            
             predictions = train_context.net.forward(batch_x, data_manager_obj = data_manager_obj, batch_y = batch_y, task_id = task_id).to(train_context.device) 
            
             accuracy = torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
            
             #avg_acc += accuracy
            
             #sub_task_accuracies[task_id] = accuracy
                 
         print("Task id ", task_id, "Validation task accuracy ", 100 * accuracy )
         
         #print("Validation sub-task accuracy: ", { task_id: 100*accuracy for task_id, accuracy in sub_task_accuracies.items() }  )
    
    
    def train(self, train_context, data_manager_obj, checkpoint_obj):

        """train model """
    
        train_context.net.train()
        
        train_accuracy, train_loss = [], []
            
        train_x = data_manager_obj.task_train_x[data_manager_obj.current_task_id]
        
        train_y = data_manager_obj.task_train_y[data_manager_obj.current_task_id]
        
        for i in range(0, train_x.shape[0], self.train_batch_size ):
            
            batch_x, batch_y = train_x[ i : i + self.train_batch_size], train_y[ i : i + self.train_batch_size]
        
            for param in train_context.net.parameters(): 
                param.grad = None   # apparently faster than optim.zero_grad()
            
            predictions = train_context.net.forward( batch_x, data_manager_obj = data_manager_obj, batch_y = batch_y, task_id = data_manager_obj.current_task_id)
            
            current_reg_loss = train_context.loss(predictions, batch_y)
            
            current_reg_loss.backward()
            
            train_context.optim.step()
        
            train_accuracy.append( torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item())
            
            train_loss.append( current_reg_loss.item())
        
            #if i%100==0:                
            
        print("task id ", data_manager_obj.current_task_id, "Train accuracy: ", 100* np.mean(train_accuracy), "Train Loss: ", np.mean(train_loss) )
                
         
            
            
    
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        checkpoint = torch.load(r"C:/Users/gauthambekal93/Research/query-only-attention-for-continual-learning/results/cifar_100/query_based_cl_V8/0/0/model_30_way_classification_20_supports.pkl" ,  map_location = train_context.device)
        
        train_context.net.load_state_dict(checkpoint["model_state"])
        
        train_context.optim.load_state_dict(checkpoint["optimizer_state"])
        
        data_manager_obj.create_test_data()
    
        
        while data_manager_obj.current_task_id < 50000: #data_manager_obj.total_tasks:
            
            start = time.perf_counter()
            
            if data_manager_obj.current_task_id > 0:
                data_manager_obj.create_task_data()
            
            #data_manager_obj.create_eval_data()
            
            self.train( train_context, data_manager_obj, checkpoint_obj)
            
            if data_manager_obj.current_task_id >=20:
                
                self.val_network(train_context, data_manager_obj, checkpoint_obj) 
                
                data_manager_obj.delete_data()
                
            
            #data_manager_obj.current_num_classes += data_manager_obj.class_increase_per_task
            
            data_manager_obj.current_task_id += 1
            
            print("Loop time ", time.perf_counter() -  start)
            
            print("===========================================================================================")
            
            self.test_network( train_context, data_manager_obj, checkpoint_obj)
            
        #checkpoint = {
        #"model_state": train_context.net.state_dict(),
        #"optimizer_state": train_context.optim.state_dict()
        #}

        #torch.save(checkpoint, r"C:/Users/gauthambekal93/Research/query-only-attention-for-continual-learning/results/cifar_100/query_based_cl_V8/0/0/model_30_way_classification_20_supports.pkl" ) 
        
        #checkpoint = torch.load(r"C:/Users/gauthambekal93/Research/query-only-attention-for-continual-learning/results/cifar_100/query_based_cl_V8/0/0/model_30_way_classification_20_supports.pkl" ,  map_location = train_context.device)
        
        #train_context.net.load_state_dict(checkpoint["model_state"])
        
        #train_context.optim.load_state_dict(checkpoint["optimizer_state"])
