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
    
    def __init__(self, input_size= 49, num_features=2000, num_outputs=1, num_hidden_layers=2, act_type='relu', opt ='adam', step_size = 0.001,
                 weight_decay=0,  momentum=0, beta_1=0.9, beta_2=0.999, loss='mse', task_datapoints = 10000, classes_per_task = 10):
            
        super().__init__()
        
        
        self.fc1 = nn.Linear(input_size, num_features)
        
        self.fc2 = nn.Linear(num_features, num_features)
        
        self.fc3 = nn.Linear(num_features, num_features)
        
        self.fc4 = nn.Linear(num_features, num_features)
        
        self.fc5 = nn.Linear(num_features, num_features)
        
        self.fc6 = nn.Linear(num_features, num_features)
        
        self.fc7 = nn.Linear(num_features, num_features)
        
        self.fc8 = nn.Linear(num_features, num_features)
        
        self.fc9 = nn.Linear(num_features, num_features)
        
        self.fc10 = nn.Linear(num_features, num_features)
        
        self.fc11 = nn.Linear(num_features, num_outputs)
        
        
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

        
    def forward(self, input_1, input_2):
        
        x = torch.cat((input_1, input_2), dim = -1)
        
        self.logits_1 = F.relu(self.fc1( x ))
        
        self.logits_2 =  F.relu(self.fc2(self.logits_1) ) 
        
        self.logits_3 =  F.relu(self.fc3(self.logits_2) ) 
        
        self.logits_4 =  F.relu( self.fc4(self.logits_3) )

        self.logits_5 =  F.relu( self.fc5(self.logits_4) )
         
        self.logits_6 =  F.relu( self.fc6(self.logits_5)  )
        
        self.logits_7 =  F.relu( self.fc7(self.logits_6))
        
        self.logits_8 =  F.relu(self.fc8(self.logits_7))
        
        self.logits_9 =  F.relu( self.fc9(self.logits_8))
        
        self.logits_10 =  F.relu( self.fc10(self.logits_9))
        
        self.logits_11 =  self.fc11(self.logits_10) 
        
        """Randomize the support set """
        if input_2.ndim ==2:
            
            rand_idx = torch.randperm(input_2.shape[0])  
            
            input_2 = input_2[rand_idx,:]
            
            self.logits_11 = self.logits_11[rand_idx, :]
            
            y_pred = self.logits_11 * input_2
            
            y_pred = torch.sum(y_pred , dim = 0).unsqueeze(dim = 0)
            
            return y_pred
            
        if input_2.ndim ==3:
            
            rand_idx = torch.randperm(input_2.shape[1])  
        
            input_2 = input_2[:, rand_idx, :]
            
            self.logits_11 = self.logits_11[:, rand_idx, : ]
        
            y_pred = self.logits_11 * input_2
            
            y_pred = torch.sum(y_pred , dim = 1)
        
            return y_pred
    
    def learn(self, X, Y ):

        
       episode_loss , accuracies = [], 0
       
       #idx = random.randint(0, self.classes_per_task -1 )
       
       #idx =  torch.randint(low=0, high = self.classes_per_task, size=(self.samples_per_batch,))  # with replacement
       
       #X , Y = [X[idx]], [Y[idx]]
       
       for x, y in zip(X, Y):
           
           input_1 = x[:, : - self.classes_per_task]
           
           input_2 = x[:, - self.classes_per_task: ]
           
           self.opt.zero_grad()
          
           y_pred = self.forward( input_1, input_2 )
       
           loss = self.loss_func(y_pred, y)
           
           loss.backward()
       
           self.opt.step()
           
           self.weight_magnitude.append( self.calculate_weight_magnitude() )
          
           episode_loss.append(loss.item())
       
           accuracies += (y_pred.argmax()==y.argmax()).float()
           
    
       episode_loss = np.mean( episode_loss )
          
       accuracies = 100* (accuracies / len(X))  #was self.classes_per_task
     
       return episode_loss, accuracies
    
    def test(self, X, Y ):
        
        episode_loss  = []
        
        X = torch.stack(X, dim = 0)
        
        Y = torch.cat(Y, dim = 0)  
        
        input_1 = X[:, :,  : - self.classes_per_task ] 
        
        input_2 = X[:, :, - self.classes_per_task: ]
        
        y_pred = self.forward( input_1, input_2 )
        
        loss = self.loss_func(y_pred, Y)
        
        episode_loss.append(loss.item())
        
        accuracies = 100* ( (y_pred.argmax(dim = 1)==Y.argmax(dim =1 )).sum().float() ) / Y.shape[0]
        
        return episode_loss, accuracies
        
        '''
        for x, y in zip(X, Y):
            
            input_1 = x[:, : - self.classes_per_task]
            
            input_2 = x[:, - self.classes_per_task: ]
            
            y_pred = self.forward( input_1, input_2 )
        
            loss = self.loss_func(y_pred, y)
           
            episode_loss.append(loss.item())
            
            accuracies += (y_pred.argmax()==y.argmax()).float()
            
             
        episode_loss = np.mean( episode_loss )
           
        accuracies = 100* (accuracies / len(X))  #was self.classes_per_task
      
        return episode_loss, accuracies
        '''
    
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
        
        
        
        
        
        
        
        