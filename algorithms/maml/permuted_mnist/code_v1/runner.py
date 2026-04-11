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
import copy

class Runner:
    
    def __init__(self, num_datapoints_per_timestep, test_batch_size, num_iterations):
        
        self.num_datapoints_per_timestep = num_datapoints_per_timestep
        self.test_batch_size = test_batch_size
        self.num_iterations = num_iterations
    
    def prequential_testing(self, train_context, data_manager_obj, batch_x, batch_y):
        
        train_context.net.eval()
        
        theta_prime = self.obtain_theta_prime(train_context, data_manager_obj, create_graph=False)
        
        with torch.no_grad():
            
            _, predictions = self.compute_on_theta_prime(train_context, theta_prime, batch_x, batch_y)
        
        accuracy = 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
        
        return accuracy
                
    
    def forward_testing(self, train_context, data_manager_obj):
        
        train_context.net.eval()
        
        batch_x, batch_y = data_manager_obj.task_test_x[data_manager_obj.current_task_id], data_manager_obj.task_test_y[ data_manager_obj.current_task_id ]
        
        theta_prime = self.obtain_theta_prime(train_context, data_manager_obj, create_graph=False)
        
        with torch.no_grad():

            _, predictions = self.compute_on_theta_prime(train_context, theta_prime, batch_x, batch_y)
        
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
                      
                     predictions = train_context.net.prediction( data_manager_obj,  batch_x)
                     
                     accuracy = 100 * torch.mean((predictions == batch_y).to(torch.float32)).item()
                     
                     avg_acc += accuracy
                     
                     sub_task_accuracies[task_id] = accuracy
  
         accuracy =  avg_acc / ( len(data_manager_obj.task_test_x.keys() ) - 1 ) 
         
         return accuracy
         
    
    def compute_on_theta_prime(self, train_context, theta_prime, x, y):
        
        """HERE WE ALSO NEED TO RETUN THE ACCURACY """
        w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = theta_prime  
        
        logits_1 = F.relu( torch.nn.functional.linear(x, w1, b1))
        
        logits_2 = F.relu( torch.nn.functional.linear(logits_1, w2, b2))
        
        logits_3 = F.relu( torch.nn.functional.linear(logits_2, w3, b3))
        
        logits_4 = F.relu( torch.nn.functional.linear(logits_3, w4, b4))
        
        y_pred = torch.nn.functional.linear(logits_4, w5, b5)
    
        theta_prime_loss = train_context.loss(y_pred, y )
        
        return theta_prime_loss, y_pred

        
    def obtain_theta_prime(self, train_context, data_manager_obj, create_graph = True):
        
        theta_prime = [p.clone() for p in train_context.net.parameters()]
        
        for _ in range(self.num_iterations):
            support_x , support_y = data_manager_obj.get_fifo_data()
            
            support_y = support_y.argmax(dim =1)
            
            loss, _ = self.compute_on_theta_prime( train_context, theta_prime, support_x , support_y  )
            
            grads = torch.autograd.grad(loss, theta_prime , create_graph=create_graph)
            
            theta_prime = [ p_prime - train_context.step_size * g for p_prime, g in zip(theta_prime, grads) ]
            
        return theta_prime
    
    
    def update_theta(self, train_context, data_manager_obj, query_x, query_y, theta_prime):
        
        loss, predictions = self.compute_on_theta_prime( train_context, theta_prime, query_x , query_y  )
        
        grads = torch.autograd.grad(loss, train_context.net.parameters() )
        
        with torch.no_grad():
            
            for p, g in zip( train_context.net.parameters(), grads):
                
                p -=  train_context.step_size * g 
            
        return loss, predictions
    
    def train(self, train_context, data_manager_obj, checkpoint_obj):

        """train model """
    
        train_context.net.train()
        
        train_accuracy, train_loss, prequential_accuracy = [], [], []
            
        train_x = data_manager_obj.task_train_x[data_manager_obj.current_task_id]
        
        train_y = data_manager_obj.task_train_y[data_manager_obj.current_task_id]
        
        
        for i in range(0, train_x.shape[0], self.num_datapoints_per_timestep ):
            
            batch_x  = train_x[ i : i + self.num_datapoints_per_timestep] 
            
            batch_y = train_y[ i : i + self.num_datapoints_per_timestep]
            
            data_manager_obj.fill_fifo_buffer( batch_x, batch_y )
            
            acc = self.prequential_testing(train_context, data_manager_obj, batch_x, batch_y)
            
            prequential_accuracy.append( acc  )
            
            train_context.net.train()
            
            for param in train_context.net.parameters(): 
                param.grad = None   # apparently faster than optim.zero_grad()
            
            theta_prime = self.obtain_theta_prime( train_context, data_manager_obj)
            
            current_reg_loss, predictions = self.update_theta(train_context, data_manager_obj, batch_x, batch_y, theta_prime)
            
            train_accuracy.append( 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)) )
            
            train_loss.append( current_reg_loss)
            
    
        train_loss= torch.stack(train_loss).mean().item()
        
        train_accuracy= torch.stack(train_accuracy).mean().item()

        prequential_accuracy = np.mean(prequential_accuracy)
        
        forward_accuracy= self.forward_testing(train_context, data_manager_obj) 
        
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
            
           

