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
        
        self.fc2 = nn.Linear(num_features * 2, 1)
    
        
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)
    
    """            
    def forward(self, data_manager_obj, query_x):
       
        '''
        x = x.expand( len(support_x) , -1)
        
        x = torch.cat([x, support_x, support_y], dim = 1 )
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        
        x =  F.softmax(x, dim = 0)
        
        x = torch.sum( x * support_y , dim =0).reshape(-1, 1)
        
        return x
        '''
        
        support_x , support_y = data_manager_obj.get_fifo_data()
        
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
        
        
        x = F.relu  ( self.fc1(x) )
        
        x = self.fc2(x) 
        
        rand_idx = torch.randperm(support_y.shape[0])  
        
        support_y = support_y[rand_idx,:]
        
        x = x[:, rand_idx, :]
    
        x = x * support_y
        
        x = x.sum(dim = 1)
    
        return x
    """
    
    def forward(self, data_manager_obj, query_x):
    
        support_x , support_y = data_manager_obj.get_fifo_data()
        
        query_x = F.relu(self.fc1(query_x))
        
        support_x = F.relu(self.fc1(support_x))
    
        #ELEMENT WISE ADDITION OF SUPPORT VECTORS FOR RELATION NETWORK
        support_x = torch.matmul(support_x.T, support_y.float()).T
    
        query_embed_shape = query_x.shape
    
        support_embed_shape = support_x.shape
    
        query_x = query_x.unsqueeze(1)
    
        query_x = query_x.expand(query_embed_shape[0], support_embed_shape[0],  query_embed_shape[1] )
    
        support_x = support_x.unsqueeze(0)
    
        support_x = support_x.expand(query_embed_shape[0], support_embed_shape[0],  query_embed_shape[1]  )
    
        x = torch.cat([query_x, support_x], dim=-1)
        
        x = torch.sigmoid( self.fc2(x) )
        
        x = x.reshape(-1, x.shape[1])
        
        return x
    