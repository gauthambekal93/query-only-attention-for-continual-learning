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
             
             for task_id in data_manager_obj.task_test_x.keys():
                 
                 batch_x, batch_y = data_manager_obj.task_test_x[ task_id ], data_manager_obj.task_test_y[ task_id ]
                 
                 predictions = train_context.net.forward(batch_x, data_manager_obj = data_manager_obj, batch_y = batch_y, task_id = task_id, data_type = "eval").to(train_context.device) 
                 
                 accuracy = torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
                 
                 avg_acc += accuracy
                 
                 sub_task_accuracies[task_id] = accuracy
               

         #print("Test task accuracy: ", 100 * (avg_acc / (data_manager_obj.current_task_id + 1 ) )  )
         print("Test task accuracy: ", 100 * (avg_acc / len(data_manager_obj.task_test_x.keys() ) ) )
         
         #print("Test sub-task accuracy: ", { task_id: 100*accuracy for task_id, accuracy in sub_task_accuracies.items() }  )
     
                     
    def val_network(self, train_context, data_manager_obj, checkpoint_obj):
         
         train_context.net.eval()

         
         with torch.no_grad():
             task_id = data_manager_obj.current_task_id - 20
             #task_id = 0
             
             batch_x, batch_y = data_manager_obj.task_val_x[ task_id ], data_manager_obj.task_val_y[ task_id ]
            
             predictions = train_context.net.forward(batch_x, data_manager_obj = data_manager_obj, batch_y = batch_y, task_id = task_id, data_type = "eval").to(train_context.device) 
            
             accuracy = torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
                             
         print("Task id ", task_id, "Validation task accuracy ", 100 * accuracy )
         
         data_manager_obj.validation_accuracy.append(100 * accuracy )

         label_accuracy = {}
         for l in torch.unique(batch_y):
             mask = torch.isin(batch_y, l)
             indices = torch.nonzero(mask).squeeze()
             label_accuracy[l.item()] = torch.mean((predictions[indices].argmax(axis=1) == batch_y[indices]).to(torch.float32)).item()
             
            
    def train(self, train_context, data_manager_obj, checkpoint_obj):

        """train model """
    
        train_context.net.train()
        
        train_accuracy, train_loss = [], []
            
        train_x = data_manager_obj.task_train_x[data_manager_obj.current_task_id]
        
        train_y = data_manager_obj.task_train_y[data_manager_obj.current_task_id]
        
        for i in range(0, train_x.shape[0], self.train_batch_size ):
            
            batch_x, batch_y = train_x[ i : i + self.train_batch_size], train_y[ i : i + self.train_batch_size]
            
            batch_x = data_manager_obj.augment_batch(batch_x)
            
            for param in train_context.net.parameters(): 
                param.grad = None   # apparently faster than optim.zero_grad()
            
            predictions = train_context.net.forward( batch_x, data_manager_obj = data_manager_obj, batch_y = batch_y, task_id = data_manager_obj.current_task_id, data_type = "train")
            
            current_reg_loss = train_context.loss(predictions, batch_y)
            
            current_reg_loss.backward()
            
            train_context.optim.step()
        
            train_accuracy.append( torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item())
            
            train_loss.append( current_reg_loss.item())
            #if i%100==0:                
            
        print("task id ", data_manager_obj.current_task_id, "Train accuracy: ", 100* np.mean(train_accuracy), "Train Loss: ", np.mean(train_loss) )
                
         
            
            
    
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        
        #checkpoint = {
        #"model_state": train_context.net.state_dict(),
        #"optimizer_state": train_context.optim.state_dict()
        #}
        
        #checkpoint = torch.load(r"C:/Users/gauthambekal93/Research/query-only-attention-for-continual-learning/results/cifar_100/query_based_cl_V10/0/0/model_5_way_classification_10_supports_x.pkl" ,  map_location = train_context.device)
        
        #train_context.net.load_state_dict(checkpoint["model_state"])
        
        #train_context.optim.load_state_dict(checkpoint["optimizer_state"])

    
        while data_manager_obj.current_task_id < 50000: #data_manager_obj.total_tasks:
            
            start = time.perf_counter()
        
            data_manager_obj.create_task_data()
            
            self.train( train_context, data_manager_obj, checkpoint_obj)
            
            if data_manager_obj.current_task_id >=20: # >=20:
                
                self.val_network(train_context, data_manager_obj, checkpoint_obj) 
                
                self.test_network( train_context, data_manager_obj, checkpoint_obj)
                
                data_manager_obj.delete_data()

            
            data_manager_obj.current_task_id += 1
            
            print("Loop time ", time.perf_counter() -  start)
            
            print("===========================================================================================")
            
           
        return    
        #checkpoint = {
        #"model_state": train_context.net.state_dict(),
        #"optimizer_state": train_context.optim.state_dict()
        #}

        #torch.save(checkpoint, r"C:/Users/gauthambekal93/Research/query-only-attention-for-continual-learning/results/cifar_100/query_based_cl_V10/0/0/model_5_way_classification_20_supports_x_3.pkl" ) 
        
        '''
        #Done at task 3690
        for param_group in train_context.optim.param_groups:
            param_group['lr'] = 0.00001
        '''
       