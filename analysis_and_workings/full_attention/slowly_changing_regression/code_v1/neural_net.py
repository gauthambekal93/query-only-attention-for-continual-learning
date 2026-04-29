# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 11:17:31 2025

@author: gauthambekal93
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
import math

class feed_forward_nn(nn.Module):
    def __init__(self, input_size, num_features, num_outputs  ):
        super(feed_forward_nn, self).__init__()
        self.num_inputs = input_size
        self.num_features = num_features
        self.num_outputs = num_outputs
        
        self.fc1 = nn.Linear(input_size + num_outputs, num_features)
        
        self.query_1 = nn.Linear(num_features, num_features)
        self.key_1 = nn.Linear(num_features, num_features)
        self.value_1 = nn.Linear(num_features, num_features)
        
        self.fc2 = nn.Linear(num_features, num_outputs)
    

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.kaiming_uniform_(self.query_1.weight, nonlinearity='linear')
        nn.init.zeros_(self.query_1.bias)
        nn.init.kaiming_uniform_(self.key_1.weight, nonlinearity='linear')
        nn.init.zeros_(self.key_1.bias)
        nn.init.kaiming_uniform_(self.value_1.weight, nonlinearity='linear')
        nn.init.zeros_(self.value_1.bias)
        
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)
                
        
    def forward(self, data_manager_obj, query_x):
       
        support_x , support_y = data_manager_obj.get_fifo_data()
        
        supports =  torch.cat([support_x, support_y], dim = 1)
        
        zeros = torch.zeros(support_y.shape[1]).unsqueeze(dim =0).to(supports.device)
        
        zeros = zeros.expand( query_x.shape[0], zeros.shape[1])
        
        queries = torch.cat( [query_x, zeros ], dim = 1 )
        
        Z = []
        for query in queries:
            query = query.unsqueeze(dim =0)
            Z.append( torch.cat([supports, query], dim =0) )
            
        Z = torch.stack (Z, dim =0) 
        
        "-------Fully Connected Layer 1 "
        embedding = self.fc1(Z) 
        
        
        "----Attention Layer 1----"
        query_output_1 = self.query_1(embedding)
   
        key_output_1 = self.key_1(embedding)
           
        value_output_1 = self.value_1(embedding)
           
        attention_matrix = F.softmax( torch.matmul(query_output_1, key_output_1.transpose(-2, -1) ) / math.sqrt( self.key_1.in_features ) , dim = -1 )
           
        output = torch.matmul( attention_matrix, value_output_1)
        
        "----Residual Connection---"
        layer1_output = embedding + output        
        
        "-------Fully Connected Layer 2 "
        predictions = self.fc2(layer1_output)
        
        predictions = predictions[:, -1, :]
        
        return predictions
    
    
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

    