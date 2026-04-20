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

        
    def obtain_theta_prime(self, train_context, data_manager_obj, support_x , support_y, create_graph = True):
    
        theta_prime = [p.clone() for p in train_context.net.parameters()]
        
        for _ in range(100): #range(self.num_train_iterations):
            
            loss, _ = self.compute_on_theta_prime( train_context, theta_prime, support_x, support_y  )
            
            grads = torch.autograd.grad(loss, theta_prime , create_graph=create_graph)
            
            theta_prime =  [ p_prime - train_context.step_size * g for p_prime, g in zip(theta_prime, grads) ]     
        
        theta_prime = [t / t.sum() for t in theta_prime]    
        
        return theta_prime
    
   
        
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        checkpoint_obj.load_experiment_checkpoint(train_context, data_manager_obj)
    
        train_context.net.eval()
        
        pair_wise_theta_primes = []
        
        distance_metric = []
        
        for task_id in range(1, 100):
            
            data_manager_obj.create_task_data(task_id)
            
            train_x = data_manager_obj.task_train_x[task_id]
            
            train_y = data_manager_obj.task_train_y[task_id]
            
            batch_x  = train_x[ : self.num_datapoints_per_timestep] 
            
            batch_y = train_y[ : self.num_datapoints_per_timestep]
            
            data_manager_obj.fill_buffer(  batch_x, batch_y )
            
            supports_x , supports_y, _, _ = data_manager_obj.get_buffer_data(task_id)
              
            theta_prime = self.obtain_theta_prime( train_context, data_manager_obj, supports_x , supports_y)
              
            pair_wise_theta_primes.append(theta_prime)  
            
            if len(pair_wise_theta_primes) ==2 :
              #distance_measure = 0
              
              theta_prime1 = torch.cat([p.reshape(-1) for p in pair_wise_theta_primes[0]])
              theta_prime2 = torch.cat([p.reshape(-1) for p in pair_wise_theta_primes[1]])
              
              '''
              for  theta_prime1, theta_prime2 in zip(pair_wise_theta_primes[0], pair_wise_theta_primes[1]):
                  distance_measure += torch.norm(theta_prime1 - theta_prime2, p=2 ).item() 
                  
              distance_metric.append( distance_measure /  len(pair_wise_theta_primes[0] ))
              '''
              
              #distance_metric.append( F.cosine_similarity(theta_prime1, theta_prime2, dim=0) )
              distance_metric.append( torch.norm(theta_prime1 - theta_prime2, p=2 ).item() ) 
              
              pair_wise_theta_primes = []
              
              data_manager_obj.delete_data(task_id)
                
        for i in range(10):
            print( "Label ", i, "distance", np.mean(distance_metric[i]))
            

            
            
           

