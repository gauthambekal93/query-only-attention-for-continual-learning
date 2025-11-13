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

class MLP(nn.Module):
    
    def __init__(self, input_features= 49, num_features=2000, num_outputs=1, act_type='relu', opt ='adam', step_size = 0.001,
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

        
    def forward(self, train_x):
        
        self.cnn_logits = self.cnn(train_x).flatten(1)
        
        self.logits_1 = F.relu(self.fc1( self.cnn_logits ))
        
        self.logits_2 =  F.relu(self.fc2(self.logits_1) ) 
        
        y_pred=  self.fc3(self.logits_2)  
        
        return y_pred
    

    def learn(self, train_x, train_y_one_hot ):

        self.opt.zero_grad()
        
        y_pred = self.forward(train_x)
        
        loss = self.loss_func(y_pred, train_y_one_hot)
        
        loss.backward()
        
        self.opt.step()
        
        episode_loss = loss
    
        accuracies = 100 * (y_pred.argmax(dim = 1)==train_y_one_hot.argmax(dim =1)).float().mean()
          
        return episode_loss, accuracies

    
    
    def test(self, data_dict_val ):
        
        test_x, test_y = data_dict_val["test_x"], data_dict_val["test_y"]
        
        y_pred = self.forward(test_x)
            
        accuracies = 100* (y_pred.argmax(dim = 1)==test_y.argmax(dim =1)).float().mean()
            
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
        
        
        
        
        
        
        
        