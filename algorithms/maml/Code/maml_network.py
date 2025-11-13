# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 20:26:36 2025

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
    def __init__(self,  inner_step_size= 0.01, outer_step_size =0.0001, loss='mse', opt='adam', beta_1=0.9, beta_2=0.999, weight_decay=0.0, to_perturb =False, momentum=0, input_feature_size= 21 * 10, hidden_feature_size= 20, task_datapoints = 10000):
        super().__init__()
        
        self.input_feature_size = input_feature_size
        
        self.fc1 = nn.Linear(self.input_feature_size, hidden_feature_size)
        
        self.fc2 = nn.Linear(hidden_feature_size, 1)
        
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

                
    def get_theta_loss(self, x, y):
        
        self.logits_1 = F.relu(self.fc1(x))
        
        y_pred =  self.fc2(self.logits_1)
        
        theta_loss = self.loss_func(y_pred, y)
        
        return theta_loss
    
    
    def get_theta_prime_loss(self, theta_prime, x, y):
        
        w1, b1, w2, b2 = theta_prime  
        
        logits_1 = F.relu( torch.nn.functional.linear(x, w1, b1))
        
        y_pred = torch.nn.functional.linear(logits_1, w2, b2)
    
        theta_prime_loss = self.loss_func(y_pred, y)
        
        return theta_prime_loss
    
    
    
    def update_theta_prime(self, theta, x, y , loop_count):
        
        theta_prime = [p.clone().requires_grad_(True) for p in theta]
        
        for _ in range(loop_count):
            
            loss =  self.get_theta_prime_loss( theta_prime, x, y)
            
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
        
        for task_support in support_dict.values():
            
            support_x, support_y  = task_support[:, :-1], task_support[:, -1:]
            
            theta_prime = self.update_theta_prime( theta, support_x, support_y , loop_count = 5)
            
            theta_primes.append(theta_prime)
        
        theta_prime_loss = []
        
        for i, theta_prime in enumerate(theta_primes):
            
            query_x, query_y = query_dict[i][:, :-1], query_dict[i][:, -1:]
            
            theta_prime_loss.append( self.get_theta_prime_loss(theta_prime, query_x, query_y) )
       
        theta_prime_loss = torch.mean(torch.stack(theta_prime_loss))
        
        meta_grads = self.get_meta_gradients(theta_prime_loss, theta)
        """We need to update grad paramter of outer_prams since we are using pytorch built in optimizer like adam. """
        
        for p, g in zip(theta, meta_grads): 
             """ .deatch() here is not necessary since diffrentiation is already done and we are zeroing 
             the gradients after step. so this for best practice and safety.  """
             
             p.grad = g.detach()  
     
        self.opt.step()
         
        self.opt.zero_grad()
        
        episode_loss = theta_prime_loss.item()
         
        return episode_loss
         
   
    def test(self,support_x, support_y, queries_x, queries_y):
        
        theta = list(self.parameters())
        
        theta_prime = self.update_theta_prime( theta, support_x, support_y , loop_count = 5)
        
        with torch.no_grad():
            theta_prime_loss = self.get_theta_prime_loss(theta_prime, queries_x, queries_y)
            
        return theta_prime_loss.item()
    
    
    def calculate_hessian(self, support_x, support_y, queries_x, queries_y):
        
        theta = list(self.parameters())
       
        theta_prime = self.update_theta_prime( theta, support_x, support_y , loop_count = 5)
        
        theta_prime_loss = self.get_theta_prime_loss(theta_prime, queries_x, queries_y)
        
        theta_last_layer = list(self.fc2.parameters())  
        
        meta_grads = self.get_meta_gradients(theta_prime_loss, theta_last_layer)
        
        grads_flat = torch.cat([g.reshape(-1) for g in meta_grads])
        
        # Compute Hessian (final layer only)
        num_params = grads_flat.numel()
        
        H = torch.zeros((num_params, num_params))
        
        try:
            for i in range(num_params):
                #print(i)
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