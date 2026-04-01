# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 11:17:31 2025

@author: gauthambekal93
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim


    
'''

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
'''    
    
class feed_forward_nn(nn.Module):
    def __init__(self, input_size, num_features, num_outputs  ):
        super(feed_forward_nn, self).__init__()
        self.num_inputs = input_size
        self.num_features = num_features
        self.num_outputs = num_outputs
        
        self.fc1 = nn.Linear(input_size, num_features)
        self.fc2 = nn.Linear(num_features, num_features)
        self.fc3 = nn.Linear(num_features, num_features)
        self.fc4 = nn.Linear(num_features, num_features)
        self.fc5 = nn.Linear(num_features, num_outputs)
    
        
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
                
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x    
    
    