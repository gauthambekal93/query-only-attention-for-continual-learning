# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:56:10 2026

@author: gauthambekal93
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math

class ERNetwork(nn.Module):
    def __init__(self, input_size, num_features, num_outputs):
        super().__init__()
        self.relu = nn.ReLU()
        
        self.fc1  = nn.Linear(input_size + num_outputs, num_features)
        
        self.query_1 = nn.Linear(num_features, num_features)
        self.key_1 = nn.Linear(num_features, num_features)
        self.value_1 = nn.Linear(num_features, num_features)
        
        self.query_2 = nn.Linear(num_features, num_features)
        self.key_2 = nn.Linear(num_features, num_features)
        self.value_2 = nn.Linear(num_features, num_features)
        
        self.query_3 = nn.Linear(num_features, num_features)
        self.key_3 = nn.Linear(num_features, num_features)
        self.value_3 = nn.Linear(num_features, num_features)
          
        self.fc2 =  nn.Linear(num_features, num_outputs)
  
    
        # Initialization
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.kaiming_uniform_(self.query_1.weight, nonlinearity='linear')
        nn.init.zeros_(self.query_1.bias)
        nn.init.kaiming_uniform_(self.key_1.weight, nonlinearity='linear')
        nn.init.zeros_(self.key_1.bias)
        nn.init.kaiming_uniform_(self.value_1.weight, nonlinearity='linear')
        nn.init.zeros_(self.value_1.bias)
        
        nn.init.kaiming_uniform_(self.query_2.weight, nonlinearity='linear')
        nn.init.zeros_(self.query_2.bias)
        nn.init.kaiming_uniform_(self.key_2.weight, nonlinearity='linear')
        nn.init.zeros_(self.key_2.bias)
        nn.init.kaiming_uniform_(self.value_2.weight, nonlinearity='linear')
        nn.init.zeros_(self.value_2.bias)
        
        nn.init.kaiming_uniform_(self.query_3.weight, nonlinearity='linear')
        nn.init.zeros_(self.query_3.bias)
        nn.init.kaiming_uniform_(self.key_3.weight, nonlinearity='linear')
        nn.init.zeros_(self.key_3.bias)
        nn.init.kaiming_uniform_(self.value_3.weight, nonlinearity='linear')
        nn.init.zeros_(self.value_3.bias)
        
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)
        
            
    def get_attention(self, query_x, query_y, support_x, support_y):
        
        supports =  torch.cat([support_x, support_y], dim = 1)
        
        zeros = torch.zeros(support_y.shape[1]).unsqueeze(dim =0).to(supports.device)
        
        zeros = zeros.expand( query_x.shape[0], zeros.shape[1])
        
        queries = torch.cat( [query_x, zeros ], dim = 1 )
        
        Z = []
        for query in queries:
            query = query.unsqueeze(dim =0)
            Z.append( torch.cat([supports, query], dim =0) )
            
        Z = torch.stack (Z, dim =0) 
        
        "-------Fully Connected Layer 1 "
        embedding = self.fc1(Z) 
        
        
        "----Attention Layer 1----"
        query_output_1 = self.query_1(embedding)
   
        key_output_1 = self.key_1(embedding)
           
        value_output_1 = self.value_1(embedding)
           
        attention_matrix = F.softmax( torch.matmul(query_output_1, key_output_1.transpose(-2, -1) ) / math.sqrt( self.key_1.in_features ) , dim = -1 )
        attention_matrix = attention_matrix.detach()
        
        '''
        normalized_attention= torch.empty (attention_matrix.shape)
        
        for i in range(len(attention_matrix)):
            normalized_attention[i] = attention_matrix[i]/ attention_matrix[i].sum()
            
        return normalized_attention
        '''
        
        label_specific_attention = {}
        for label in range(10):
            idx = torch.where( query_y == label)[0][0].item()
            normalized_attention = attention_matrix[idx]/ attention_matrix[idx].sum().item()
            label_specific_attention[label] = normalized_attention[-1][: -1]
            
        return label_specific_attention
        


    
    def prediction(self, data_manager_obj, query_x, query_y = None ):
        
        support_x , support_y = data_manager_obj.get_fifo_data()
        
        #label_specific_attention =  self.get_attention(query_x, query_y, support_x, support_y )
        
        #return label_specific_attention
     
        label_specific_attention =  self.get_attention(query_x, query_y, support_x, support_y )
        
        return label_specific_attention
    
    
    
    