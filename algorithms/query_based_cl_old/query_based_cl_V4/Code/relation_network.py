# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:46:47 2025

@author: gauthambekal93
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch
import numpy as np

class RN(nn.Module):
    def __init__(self, step_size= 0.0001, loss='mse', opt='adam', beta_1=0.9, beta_2=0.999, weight_decay=0.0, to_perturb =False, momentum=0, input_feature_size= 21 * 10, task_datapoints = 10000):
        super().__init__()
        
        self.input_feature_size = input_feature_size
        
        self.fc1 = nn.Linear(self.input_feature_size, 20)
        
        self.fc2 = nn.Linear(20, 1)
    
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        if opt == 'sgd':
            self.opt = optim.SGD(self.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.parameters(), lr=step_size)
            #self.opt = optim.Adam(self.parameters(), lr=step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.parameters(), lr=step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
            
        self.loss = loss
        
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]
        
        self.logits_1 = None
        
        self.activation_outputs = torch.ones( task_datapoints, self.fc1.out_features  )
        
        self.datapoint_count = 0
        
        self.weight_magnitude = []
        
    def forward(self, x):
        
        x_support_query = x[:, :-1] 
        
        y_support = x[:, -1:]
        
        self.logits_1 = F.relu(self.fc1(x_support_query))
        
        #self.logits_1 = self.fc1(x_support_query)
        
        self.logits_2 =  self.fc2(self.logits_1)
        
        self.logits_3 =  F.softmax(self.logits_2, dim = 0)
        
        y_pred = torch.sum( self.logits_3 * y_support , dim =0)
        
        return y_pred
    
    
    def learn(self, x, y ):
        
            
        self.opt.zero_grad()
       
        y_pred = self.forward( x )
        
        y = y.reshape(-1)
        
        loss = self.loss_func(y_pred, y)
      
        loss.backward()
    
        self.opt.step()
    
        #self.count_dead_units()

        self.weight_magnitude.append( self.calculate_weight_magnitude() )
       
        episode_loss = loss.item()
    
        return episode_loss
        
    
    def test(self, X, Y ):
            
        episode_loss = []
        
        for x, y in zip(X, Y):
                
                y_pred = self.forward( x )
            
                loss = self.loss_func(y_pred, y)
               
                episode_loss.append(loss.item())
            
        episode_loss = np.mean( episode_loss )
            
        return episode_loss
    
    
    def calculate_hessian(self,X, Y):
        
        params = list(self.fc2.parameters() )
        
        loss = []
        
        for x, y in zip(X, Y):
            
            y_pred = self.forward( x )
        
            loss.append( self.loss_func(y_pred, y) )
    
        loss = torch.stack(loss).mean()
        
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        grads_flat = torch.cat([g.reshape(-1) for g in grads])
        
        # Compute Hessian (final layer only)
        num_params = grads_flat.numel()
        
        H = torch.zeros((num_params, num_params))
        
        for i in range(num_params):
            #print(i)
            second_grads = torch.autograd.grad(grads_flat[i], params, retain_graph=True)
            
            H[i] = torch.cat([g.reshape(-1) for g in second_grads]).detach()
        
        # Effective rank of Hessian
        #eigenvalues = torch.linalg.eigvalsh(H)
        
        try:
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
   
        
        
#torch.save(self.state_dict(), 'MML_slowly_changing_regression.pth')       
        
        
        
        
        
        
        
        