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
   
    def initialize_fisher(self):
        self.prev_params = {}
        self.fisher = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                self.prev_params[name] = p.detach().clone()
                self.fisher[name] = torch.zeros_like(p)

    def update_fisher(self, x, y, alpha=0.9):
        self.zero_grad()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        for name, p in self.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.fisher[name] = alpha * self.fisher[name] + (1 - alpha) * (p.grad.detach() ** 2)
    
    def update_prev_params(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                self.prev_params[name] = p.detach().clone()
    
    def ewc_loss(self):
        loss = 0.0
        for name, p in self.named_parameters():
            if p.requires_grad:
                loss = loss + (self.fisher[name] * (p - self.prev_params[name]) ** 2).sum()
        return loss
