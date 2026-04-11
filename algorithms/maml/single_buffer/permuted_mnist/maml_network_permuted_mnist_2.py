# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 13:05:20 2025

@author: gauthambekal93
"""



import os
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch
import numpy as np
from collections import deque
import time
import random


class MAML(nn.Module):
    
    def __init__(self, inner_step_size= 0.01, outer_step_size =0.0001, input_size= 49, num_features=2000, num_outputs=1, num_hidden_layers=2, act_type='relu', opt ='adam', step_size = 0.001, weight_decay=0,  momentum=0, beta_1=0.9, beta_2=0.999, loss='mse', task_datapoints = 10000, classes_per_task = 10,no_of_task_sampled=20):
            
        super().__init__()
        
        self.classes_per_task = classes_per_task
        
        self.fc1 = nn.Linear(input_size, num_features)
        
        self.fc2 = nn.Linear(num_features, num_features)
        
        self.fc3 = nn.Linear(num_features, num_features)
        
        self.fc4 = nn.Linear(num_features, num_features)
        
        self.fc5 = nn.Linear(num_features, self.classes_per_task)
        
    
        self.inner_step_size = inner_step_size
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        if opt == 'sgd':
            self.opt = optim.SGD(self.parameters(), lr=outer_step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.parameters(), lr= outer_step_size)
            #self.opt = optim.Adam(self.parameters(), lr=step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.parameters(), lr=outer_step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
            
        self.loss = loss
        
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]
        
        self.logits_1 = None
        
        self.activation_outputs = torch.ones( task_datapoints, self.fc1.out_features  )
        
        self.datapoint_count = 0
        
        self.weight_magnitude = []
   
        self.no_of_task_sampled = no_of_task_sampled
    
    def get_theta_prime_loss(self, theta_prime, x, y):
        
        """HERE WE ALSO NEED TO RETUN THE ACCURACY """
        w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = theta_prime  
        
        logits_1 = F.relu( torch.nn.functional.linear(x, w1, b1))
        
        logits_2 = F.relu( torch.nn.functional.linear(logits_1, w2, b2))
        
        logits_3 = F.relu( torch.nn.functional.linear(logits_2, w3, b3))
        
        logits_4 = F.relu( torch.nn.functional.linear(logits_3, w4, b4))
        
        y_pred = torch.nn.functional.linear(logits_4, w5, b5)
    
        theta_prime_loss = self.loss_func(y_pred, y)
        
        accuracies = 100 * ((y_pred.argmax(dim = 1) == y.argmax(dim = 1) ).float().sum() / len(y) )
       
        return theta_prime_loss, accuracies
    
    
    
    def update_theta_prime(self, theta, x, y , loop_count):
        
        theta_prime = [p.clone().requires_grad_(True) for p in theta]
        
        for _ in range(loop_count):
            
            loss, _ =  self.get_theta_prime_loss( theta_prime, x, y)
            
            grads = torch.autograd.grad(loss, theta_prime, create_graph=True)  
            
            theta_prime = [ theta_prime[i] - self.inner_step_size * grads[i] for i in range(len(theta_prime)) ]
        
        return theta_prime
        
        
    
    def get_meta_gradients(self, theta_prime_loss, theta):
        
        """meta_params used to obtain prediction and hence loss. Then diffrentiate loss wrt meta_params. 
        The create_graph=True for calculating the hessian. So that meta_grads have the computational graph"""
        meta_grads = torch.autograd.grad(theta_prime_loss, theta, create_graph=True)
        
        return meta_grads
    

        
    def learn(self, support_dict, query_dict):
        
        theta = list(self.parameters())
        
        theta_primes = []
        
        random_task_indices = list(support_dict.keys())[:self.no_of_task_sampled]
        
        for i, index_val in enumerate(random_task_indices):
            
            task_support = support_dict[index_val]
            
            support_x, support_y  = task_support[:, : -(self.classes_per_task+ 1)], task_support[:,  -(self.classes_per_task):]
            
            theta_prime = self.update_theta_prime( theta, support_x, support_y , loop_count = 5)
            
            theta_primes.append(theta_prime)
        
        theta_prime_loss, accuracies = [], 0
        
        for i, index_val in enumerate(random_task_indices):
        
            theta_prime = theta_primes[i]
            
            query_x, query_y = query_dict[index_val][:, : -(self.classes_per_task+ 1)], query_dict[i][:,  -(self.classes_per_task):]
            
            loss, accuracy = self.get_theta_prime_loss(theta_prime, query_x, query_y) 
            
            theta_prime_loss.append(loss )
            
            accuracies += accuracy
        
        accuracies = accuracies/ len(theta_primes)
        
        theta_prime_loss = torch.mean(torch.stack(theta_prime_loss))
        
        meta_grads = self.get_meta_gradients(theta_prime_loss, theta)
        """We need to update grad paramter of outer_prams since we are using pytorch built in optimizer like adam. """
        
        for p, g in zip(theta, meta_grads): 
             """ .deatch() here is not necessary since diffrentiation is already done and we are zeroing 
             the gradients after step. so this for best practice and safety.  """
             
             p.grad = g.detach()  
     
        self.opt.step()
         
        self.opt.zero_grad()
        
        return theta_prime_loss.item(), accuracies
         
   
    def test(self,support_x, support_y, queries_x, queries_y):
        
        theta = list(self.parameters())
        
        theta_prime = self.update_theta_prime( theta, support_x, support_y , loop_count = 5)
        
        with torch.no_grad():
            theta_prime_loss, accuracies = self.get_theta_prime_loss(theta_prime, queries_x, queries_y)
            
        return theta_prime_loss.item(), accuracies
    
    
    def calculate_hessian(self, support_x, support_y, queries_x, queries_y):
        
        theta = list(self.parameters())
       
        theta_prime = self.update_theta_prime( theta, support_x, support_y , loop_count = 5)
        
        theta_prime_loss, _ = self.get_theta_prime_loss(theta_prime, queries_x, queries_y)
        
        theta_last_layer = list(self.fc5.parameters())  
        
        meta_grads = self.get_meta_gradients(theta_prime_loss, theta_last_layer)
        
        grads_flat = torch.cat([g.reshape(-1) for g in meta_grads])
        
        # Compute Hessian (final layer only)
        num_params = grads_flat.numel()
        
        H = torch.zeros((num_params, num_params))
        
        try:
           for i in range(num_params):
                
                second_grads = torch.autograd.grad(grads_flat[i], theta_last_layer, retain_graph=True)
                
                H[i] = torch.cat([g.reshape(-1) for g in second_grads]).detach()
                
                epsilon = 1e-6
                H_stable = H + epsilon * torch.eye(H.shape[0])
                eigenvalues = torch.linalg.eigvalsh(H_stable)
        except:
            print("stop")
            print("stop")
            print("stop")
            
        eigenvalues = torch.clamp(eigenvalues, min=1e-12)  # Prevent log(0)
        
        p = eigenvalues / eigenvalues.sum()
        
        entropy = -torch.sum(p * torch.log(p))
        
        effective_rank = torch.exp(entropy)
        
        return effective_rank
    
        
    
    def count_dead_units(self):
        
        zero_op_indices= torch.where(self.logits_1 == 0)[0]
        
        self.activation_outputs[self.datapoint_count, zero_op_indices ] = 0 
        
        self.datapoint_count += 1
   
   
    def calculate_weight_magnitude(self):
        total = 0
        count = 0
        for name, param in self.named_parameters():
             total += param.abs().sum().item()
             count += param.numel()
        return total / count