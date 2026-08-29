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
                
    def forward(self, data_manager_obj, query_x):
               
        support_x , support_y = data_manager_obj.get_fifo_data( )
        
        support_embedding =  F.relu  ( self.fc1(support_x) )
        support_embedding =  self.fc2(support_embedding) 

        query_embedding = F.relu  ( self.fc1(query_x) )
        query_embedding = self.fc2(query_embedding) 

        query_embed_shape = query_embedding.shape 
        support_embed_shape = support_embedding.shape

        query_embedding = query_embedding.unsqueeze(1)

        query_embedding = query_embedding.expand(query_embed_shape[0], support_embed_shape[0],  query_embed_shape[1] )

        support_embedding = support_embedding.unsqueeze(0)

        support_embedding = support_embedding.expand(query_embed_shape[0], support_embed_shape[0],  query_embed_shape[1]  )
   
        similarities = F.cosine_similarity(
        query_embedding,       
        support_embedding,    
        dim=-1
        )
   
        attention = torch.softmax(similarities, dim=1)
       
        predictions = torch.matmul(attention, support_y.float() )
       
        return predictions


        
    
    