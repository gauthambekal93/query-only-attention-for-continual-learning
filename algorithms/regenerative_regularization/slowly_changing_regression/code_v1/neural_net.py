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
    
    def fill_initial_params(self):
        self.init_params = {}
        for name, p in self.named_parameters():
            self.init_params[name] = p.detach().clone()


    def regenerative_loss(self):
        loss = 0.0
        for name, p in self.named_parameters():
            loss += ((p - self.init_params[name]) ** 2).sum()
        return loss

            
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


    