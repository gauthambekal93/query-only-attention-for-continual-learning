# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:56:10 2026

@author: gauthambekal93
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class ERNetwork(nn.Module):
    def __init__(self, input_size, num_features, num_outputs):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1  = nn.Linear(input_size, num_features)
        self.fc2 =  nn.Linear(num_features, num_features)
        self.fc3 =  nn.Linear(num_features, num_features)
        self.fc4 =  nn.Linear(num_features, num_features)
        self.fc5 =  nn.Linear(num_features, num_outputs)
        
        # Initialization
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc3.bias)

        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc4.bias)

        nn.init.kaiming_uniform_(self.fc5.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc5.bias)

        
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

            
    def classify_images(self, x):
        x = self.relu  ( self.fc1(x) )
        x = self.relu ( self.fc2(x) )
        x = self.relu  ( self.fc3(x)) 
        x = self.relu (self.fc4(x))
        x = self.fc5(x)
        return x
    
    def forward(self, x):
        return self.classify_images(x)
    
    