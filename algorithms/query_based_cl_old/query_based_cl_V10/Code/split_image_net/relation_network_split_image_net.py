# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 21:16:32 2025

@author: gauthambekal93
"""

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
import time 
import random

class RN(nn.Module):
    
    def __init__(self, input_features= 49, num_features=2000, num_outputs=1, num_hidden_layers=2, act_type='relu', opt ='adam', step_size = 0.001,
                 weight_decay=0,  momentum=0, beta_1=0.9, beta_2=0.999, loss='mse', task_datapoints = 10000, classes_per_task = 10):
            
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3,  32,  5, padding=1), nn.ReLU(), nn.MaxPool2d(2),  
            nn.Conv2d(32, 64,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(input_features , num_features)
        
        self.fc2 = nn.Linear(num_features, num_features)
        
        self.fc3 = nn.Linear(num_features, num_outputs)
        
        self.classes_per_task = classes_per_task
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        if opt == 'sgd':
            self.opt = optim.SGD(self.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.parameters(), lr=step_size)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.parameters(), lr=step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
            
        self.loss = loss
        
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]
        
        self.logits_1 = None
        
        self.activation_outputs = torch.ones( task_datapoints, self.fc1.out_features  )
        
        self.datapoint_count = 0
        
        self.weight_magnitude = []

        
    def forward(self, query_x, support_x, support_y):
        
        query_x = self.cnn(query_x).flatten(1)
        
        support_x = self.cnn(support_x).flatten(1)
        
        x = torch.cat( (query_x, support_x), dim = -1)
        
        self.logits_1 = F.relu(self.fc1( x ))
        
        self.logits_2 =  F.relu(self.fc2(self.logits_1) ) 
        
        self.logits_3 =  self.fc3(self.logits_2)  
        
        """Randomize the support set """
        rand_idx = torch.randperm(support_y.shape[0])  
        
        support_y = support_y[rand_idx,:]
        
        self.logits_3 = self.logits_3[rand_idx, :]
        
        y_pred = self.logits_3 * support_y
        
        y_pred = torch.sum(y_pred , dim = 0).unsqueeze(dim = 0)
        
        return y_pred
    

    def learn(self, data_dict ):
 
       episode_loss , accuracies = [], 0
       
       for query_x, query_y, support_x, support_y in zip(data_dict["query_x"], data_dict["query_y"], data_dict["support_x"], data_dict["support_y"]):
           
           self.opt.zero_grad()
           
           y_pred = self.forward(query_x, support_x, support_y)
           
           loss = self.loss_func(y_pred, query_y)
           
           loss.backward()
           
           self.opt.step()
           
           episode_loss.append(loss.item())
       
           accuracies += (y_pred.argmax()==query_y.argmax()).float()
           
       episode_loss = np.mean( episode_loss )
          
       accuracies = 100* (accuracies / len(data_dict["query_x"] ))
     
       return episode_loss, accuracies

    
    
    def test(self, data_dict_val ):
        
        accuracies = 0
        
        for query_x, query_y, support_x, support_y in zip(data_dict_val["query_x"], data_dict_val["query_y"], data_dict_val["support_x"], data_dict_val["support_y"]):
            
            y_pred = self.forward(query_x, support_x, support_y)
            
            accuracies += (y_pred.argmax()==query_y.argmax()).float()
            
        accuracies = 100* (accuracies / len(data_dict_val["query_x"] ))
      
        return accuracies
        
    
    def calculate_hessian(self,X, Y):
        
        params = list(self.fc11.parameters() )
        
        loss = []
        
        for x, y in zip(X, Y):
            
            input_1 = x[:, : - self.classes_per_task]
            
            input_2 = x[:, - self.classes_per_task: ]
           
            y_pred = self.forward( input_1, input_2 )
        
            loss.append( self.loss_func(y_pred, y) )
    
        loss = torch.stack(loss).mean()
        
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        grads_flat = torch.cat([g.reshape(-1) for g in grads])
        
        # Compute Hessian (final layer only)
        num_params = grads_flat.numel()
        
        H = torch.zeros((num_params, num_params))
        
        for i in range(num_params):
            
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
        
        
        
        
        
        
        
        