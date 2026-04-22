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
import random

class Runner:
    
    def __init__(self, num_datapoints_per_timestep, test_batch_size, num_train_iterations, num_tasks_per_update):
        
        self.num_datapoints_per_timestep = num_datapoints_per_timestep
        self.test_batch_size = test_batch_size
        self.num_train_iterations = num_train_iterations
        self.num_tasks_per_update = num_tasks_per_update
        
        
    def prequential_testing(self, train_context, data_manager_obj, batch_x, batch_y):
        
        train_context.net.eval()
        
        supports_x, supports_y, _ , _ = data_manager_obj.get_buffer_data( np.array([  data_manager_obj.buffer_key ]) )
        
        theta_prime = self.obtain_theta_prime(train_context, data_manager_obj, supports_x , supports_y, create_graph=False)
        
        with torch.no_grad():
            
            _, predictions = self.compute_on_theta_prime(train_context, theta_prime[data_manager_obj.buffer_key ], batch_x, batch_y)
        
        accuracy = 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
        
        return accuracy
                
    
    def forward_testing(self, train_context, data_manager_obj):
        
        train_context.net.eval()
        
        batch_x, batch_y = data_manager_obj.task_test_x[data_manager_obj.current_task_id], data_manager_obj.task_test_y[ data_manager_obj.current_task_id ]
        
        supports_x, supports_y, _ , _ = data_manager_obj.get_buffer_data( [  data_manager_obj.buffer_key ] )
        
        theta_prime = self.obtain_theta_prime(train_context, data_manager_obj, supports_x , supports_y, create_graph=False)
    
        with torch.no_grad():

            _, predictions = self.compute_on_theta_prime(train_context, theta_prime[data_manager_obj.buffer_key ], batch_x, batch_y)
        
        accuracy = 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
        
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

        
    def obtain_theta_prime(self, train_context, data_manager_obj, supports_x , supports_y, create_graph = True):
        
        theta_primes = {}
        
        for task_id in supports_x.keys():
            
            theta_primes[task_id] = [p.clone() for p in train_context.net.parameters()]
            
            for _ in range(self.num_train_iterations):
                
                loss, _ = self.compute_on_theta_prime( train_context, theta_primes[task_id], supports_x[task_id], supports_y[task_id]  )
                
                grads = torch.autograd.grad(loss, theta_primes[task_id] , create_graph=create_graph)
                
                theta_primes[task_id] =  [ p_prime - train_context.inner_step_size * g for p_prime, g in zip(theta_primes[task_id], grads) ]     
            
        return theta_primes
    
    
    def update_theta(self, train_context, data_manager_obj, theta_primes, queries_x, queries_y):
        
        theta_prime_loss, train_accuracies = [], []
         
        for task_id in queries_x.keys():
            
             loss, predictions = self.compute_on_theta_prime( train_context, theta_primes[task_id], queries_x[task_id] , queries_y[task_id]  )
             
             train_accuracies.append( 100 * torch.mean((predictions.argmax(axis=1) == queries_y[task_id] ).to(torch.float32)) ) 
             
             theta_prime_loss.append(loss)
        
        theta_prime_loss = torch.mean(torch.stack(theta_prime_loss))
        
        train_accuracies = torch.mean(torch.stack ( train_accuracies))
        
        grads = torch.autograd.grad(theta_prime_loss, train_context.net.parameters() )
    
        for p, g in zip( train_context.net.parameters(), grads): 
 
             p.grad = g.detach()  
        
        train_context.opt.step()
         
        train_context.opt.zero_grad()
        
        return theta_prime_loss, train_accuracies
    
    
    
        
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        checkpoint_obj.load_experiment_checkpoint(train_context, data_manager_obj)
        
        while data_manager_obj.current_task_id < data_manager_obj.num_tasks_in_buffer :
                
                data_manager_obj.create_task_data()
                
                train_x = data_manager_obj.task_train_x[data_manager_obj.current_task_id]
                
                train_y = data_manager_obj.task_train_y[data_manager_obj.current_task_id]
                
                data_manager_obj.fill_buffer(  train_x ,  train_y )
                
                data_manager_obj.current_task_id += 1
            
        
        train_accuracy, train_loss = [], []    
        
        train_context.net.train()
        
        for epoch in range(100000):
            
            selected_task_ids = np.random.permutation(data_manager_obj.num_tasks_in_buffer)[:self.num_tasks_per_update]
            
            supports_x , supports_y, queries_x, queries_y = data_manager_obj.get_buffer_data(selected_task_ids)
            
            theta_primes = self.obtain_theta_prime( train_context, data_manager_obj, supports_x , supports_y)
        
            current_reg_loss, acc = self.update_theta(train_context, data_manager_obj, theta_primes, queries_x, queries_y )
        
            train_loss.append( current_reg_loss.item())
        
            train_accuracy.append( acc.item() )
            
            if epoch %100==0:
                print("Epoch ", epoch, "Train loss ",np.mean(train_loss) , "Train acc ",np.mean(train_accuracy))
                
                checkpoint_obj.save_model_checkpoint( train_context, data_manager_obj, np.mean(train_loss), data_manager_obj.current_task_id)

        
        
        print("===========================================================================================")
        
           

