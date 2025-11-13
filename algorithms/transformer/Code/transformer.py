# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:46:23 2025

@author: gauthambekal93
"""

import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
from torch import optim

import random
import numpy as np

torch.manual_seed(20)
np.random.seed(20)
random.seed(20)

class Transformers(nn.Module):
    def __init__(self, step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0, to_perturb =False, momentum=0, feature_size= 21, task_datapoints = 10000 ):
        super().__init__()
        self.feature_size = feature_size
        self.query_1 = nn.Linear(feature_size, feature_size)
        self.key_1 = nn.Linear(feature_size, feature_size)
        self.value_1 = nn.Linear(feature_size, feature_size)
        
        self.query_2 = nn.Linear(feature_size, feature_size)
        self.key_2 = nn.Linear(feature_size, feature_size)
        self.value_2 = nn.Linear(feature_size, feature_size)
    
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        self.to_perturb = to_perturb
        
        if opt == 'sgd':
            self.opt = optim.SGD(self.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.parameters(), lr=step_size)
            #self.opt = optim.Adam(self.parameters(), lr=step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.parameters(), lr=step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
        
        self.loss = loss
        
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]
        
        self.weight_magnitude = []
        
        
    def forward(self, Z):
        
        #Z = Z.T #THIS MAY NEED TO BE COMMENTED BASED ON SHAPE OF Z !!!
        
        "----Layer 1----"
        query_output_1 = self.query_1(Z)
        
        key_output_1 = self.key_1(Z)
        
        value_output_1 = self.value_1(Z)
        
        attention_matrix = F.softmax( torch.matmul(query_output_1, key_output_1. T ) / math.sqrt( self.key_1.in_features ) , dim = 1 )
        
        output = torch.matmul( attention_matrix, value_output_1)
        
        "----Residual Connection---"
        layer1_output = Z + output
        
        "----Layer 2----"
        query_output_2 = self.query_2(layer1_output)
        
        key_output_2 = self.key_2(layer1_output)
        
        value_output_2 = self.value_2(layer1_output)
        
        attention_matrix = F.softmax( torch.matmul(query_output_2, key_output_2. T ) / math.sqrt( self.key_2.in_features ) , dim = 1 )
        
        output = torch.matmul( attention_matrix, value_output_2)
        
        layer2_output =  layer1_output + output
        
        y_pred = layer2_output[-1:, -1:]
        
        return y_pred
    
    
    def learn(self, z, y):
        """
        Learn using one step of gradient-descent
        :param x: input
        :param target: desired output
        :return: loss
        """
        
        self.opt.zero_grad()
        
        y_pred = self.forward(z)
        
        loss = self.loss_func(y_pred, y)

        loss.backward()
        
        self.opt.step()
        
        #self.weight_magnitude.append( self.calculate_weight_magnitude() )
        
        if self.to_perturb:
            self.perturb()
        if self.loss == 'nll':
            return loss.detach(), y_pred.detach()
        
        return loss.detach()
   
    
    def test(self, Z, Y ):
            
        episode_loss = []
        
        for z, y in zip(Z, Y):
                
                y_pred = self.forward( z )
            
                loss = self.loss_func(y_pred,  y.expand( y_pred.shape) )
               
                episode_loss.append(loss.item())
            
        episode_loss = np.mean( episode_loss )
            
        return episode_loss
    
    
    def calculate_hessian(self,Z, Y):
        
        #params = [ list(self.query_2.parameters() )[0][-1:,:], list(self.query_2.parameters() )[1][-1:] ] 
        params = list(self.query_2.parameters() )
        
        loss = []
        
        for z, y in zip(Z, Y):
            
            y_pred = self.forward( z )
        
            loss.append( self.loss_func(y_pred,  y.expand( y_pred.shape) ) )
    
        loss = torch.stack(loss).mean()
        
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # last row of weight and  last element of bias
        grads_flat = torch.cat([ grads[0][-1, :].reshape(-1), grads[1][-1:].reshape(-1) ])
        
        # Compute Hessian (final layer only)
        num_params = grads_flat.numel()
        
        H = torch.zeros((num_params, num_params))
        
        for i in range(num_params):
        
            second_grads = torch.autograd.grad(grads_flat[i], params, retain_graph=True)
        
            H[i] = torch.cat([ second_grads[0][-1, :].reshape(-1), second_grads[1][-1:].reshape(-1) ]).detach()
    
        
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
    
    
    def calculate_weight_magnitude(self):
           total = 0
           count = 0
           for name, param in self.named_parameters():
                #print("Param name ", name)
                total += param.abs().sum().item()
                count += param.numel()
           return total / count
           
        