# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 11:17:31 2025

@author: gauthambekal93
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim


    


class feed_forward_nn(nn.Module):
    def __init__(self, input_size, num_features, num_outputs  ):
        super(feed_forward_nn, self).__init__()
        self.num_inputs = input_size
        self.num_features = num_features
        self.num_outputs = num_outputs
        
        self.fc1 = nn.Linear(input_size, num_features)
        
        self.fc2 = nn.Linear(num_features, num_outputs)
    
        
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)
                
    def forward(self, data_manager_obj, query_x):
       
        support_x , support_y = data_manager_obj.get_fifo_data( )
        
        query_shape = query_x.shape 
           
        support_shape = support_x.shape
           
        query_x = query_x.unsqueeze(1)
           
        query_x = query_x.expand(query_shape[0], support_shape[0],  query_shape[1] )
           
        support_x = support_x.unsqueeze(0)
           
        support_x = support_x.expand(query_shape[0], support_shape[0],  query_shape[1]  )
           
        x = torch.cat([query_x + support_x, torch.abs(query_x - support_x)], dim=-1)
           
        temp_y = support_y.unsqueeze(dim = 0)
           
        temp_y = temp_y.expand(x.shape[0], temp_y.shape[1], temp_y.shape[2])
           
        x = torch.cat( [x, temp_y], dim = 2)
        
        
        x = F.relu  ( self.fc1(x) )
        
        x = self.fc2(x) 
        
        rand_idx = torch.randperm(support_y.shape[0])  
        
        support_y = support_y[rand_idx,:]
        
        x = x[:, rand_idx, :]
    
        x = x * support_y
        
        x = x.sum(dim = 1)
    
        return x
    
    
    def calculate_effective_rank(self,X, Y, data_manager_obj, loss_func):
        
        params = params = list(self.parameters()) #list(self.fc2.parameters() )
        
        y_pred = self.forward( data_manager_obj, X )
    
        loss = loss_func(y_pred, Y)

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

            
        eigenvalues = torch.clamp(eigenvalues, min=1e-12)  # Prevent log(0)
        
        p = eigenvalues / eigenvalues.sum()
        
        entropy = -torch.sum(p * torch.log(p))
        
        effective_rank = torch.exp(entropy).item()
        
        return effective_rank
    
    
    