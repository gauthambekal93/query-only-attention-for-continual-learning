# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:56:10 2026

@author: gauthambekal93
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)
    
    
class ERNetwork(nn.Module):
    def __init__(self, input_size, num_features, num_outputs):
        super().__init__()
        self.crelu = CReLU()
        self.fc1  = nn.Linear(input_size, num_features)
        self.fc2 =  nn.Linear(num_features *2 , num_features)
        self.fc3 =  nn.Linear(num_features *2 , num_features)
        self.fc4 =  nn.Linear(num_features *2 , num_features)
        self.fc5 =  nn.Linear(num_features *2 , num_outputs)
        
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

        
        
        
            
    def classify_images(self, x):
        x = self.crelu  ( self.fc1(x) )
        x = self.crelu ( self.fc2(x) )
        x = self.crelu  ( self.fc3(x)) 
        x = self.crelu (self.fc4(x))
        x = self.fc5(x)
        return x
    
    def forward(self, x):
        return self.classify_images(x)
    
    