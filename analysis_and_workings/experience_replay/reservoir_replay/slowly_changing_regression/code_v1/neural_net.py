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
                
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def calculate_effective_rank(self,X, Y, loss_func):
     
     params = params = list(self.parameters()) #list(self.fc2.parameters() )
     
     y_pred = self.forward( X )
 
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

    