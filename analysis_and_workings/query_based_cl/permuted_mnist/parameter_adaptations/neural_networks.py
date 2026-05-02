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
        self.fc1  = nn.Linear(input_size * 2 + num_outputs, num_features)
        self.fc2 =  nn.Linear(num_features, num_features)
        self.fc3 =  nn.Linear(num_features, num_features)
        self.fc4 =  nn.Linear(num_features, num_features)
        self.fc5 =  nn.Linear(num_features, 1)
        
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

        
            
    def classify_images(self, query_x, query_y, support_x, support_y):
        
        query_shape = query_x.shape 
        
        support_shape = support_x.shape
        
        query_x = query_x.unsqueeze(1)
        
        query_x = query_x.expand(query_shape[0], support_shape[0],  query_shape[1] )
        
        support_x = support_x.unsqueeze(0)
        
        support_x = support_x.expand(query_shape[0], support_shape[0],  query_shape[1]  )
        
        x = torch.cat([query_x + support_x, torch.abs(query_x - support_x)], dim=-1)
        
        temp_y = support_y.unsqueeze(dim = 0)
        
        temp_y = temp_y.expand(x.shape[0], temp_y.shape[1], temp_y.shape[2])
        
        x = torch.cat( [x, temp_y], dim = 2)
        
        x = self.relu  ( self.fc1(x) )
        x = self.relu ( self.fc2(x) )
        x = self.relu  ( self.fc3(x)) 
        x = self.relu (self.fc4(x))
        x = self.fc5(x)
        
        rand_idx = torch.randperm(support_y.shape[0])  
        
        support_y = support_y[rand_idx,:]
        
        x = x[:, rand_idx, :]
    
        x = x * support_y
        
        x = x.sum(dim = 1)

        return x
    

    
    def prediction(self, data_manager_obj, query_x, query_y):
        
        support_x , support_y = data_manager_obj.get_fifo_data( )
        x = self.classify_images(query_x, query_y, support_x, support_y )
    
        return x

    
    
    
    
    