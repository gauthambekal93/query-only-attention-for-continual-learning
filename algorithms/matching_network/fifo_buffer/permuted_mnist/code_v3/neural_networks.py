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
        self.fc1  = nn.Linear(input_size , num_features)
        self.fc2 =  nn.Linear(num_features, num_features)
        self.fc3 =  nn.Linear(num_features, num_features)
        self.fc4 =  nn.Linear(num_features , num_features)
        self.fc5 =  nn.Linear(num_features , 60)
        
        
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

        
    def classify_images(self, query_x, support_x, support_y):
        
        support_embedding = self.relu ( self.fc1(support_x))
        support_embedding = self.relu (self.fc2(support_embedding))
        support_embedding = self.relu (self.fc3(support_embedding))
        support_embedding = self.relu (self.fc4(support_embedding))
        support_embedding = self.fc5(support_embedding)
        
        query_embedding = self.relu (self.fc1(query_x))
        query_embedding = self.relu (self.fc2(query_embedding))
        query_embedding = self.relu (self.fc3(query_embedding))
        query_embedding = self.relu (self.fc4(query_embedding))
        query_embedding = self.fc5(query_embedding)
        
        
        query_embed_shape = query_embedding.shape 
        
        support_embed_shape = support_embedding.shape
        
        query_embedding = query_embedding.unsqueeze(1)
        
        query_embedding = query_embedding.expand(query_embed_shape[0], support_embed_shape[0],  query_embed_shape[1] )
        
        support_embedding = support_embedding.unsqueeze(0)
        
        support_embedding = support_embedding.expand(query_embed_shape[0], support_embed_shape[0],  query_embed_shape[1]  )
        
        similarities = F.cosine_similarity(    #[B, S]
        query_embedding,       # [B, 1, D]
        support_embedding,    # [1, S, D]
        dim=-1
        )
        
        attention = torch.softmax(similarities, dim=1)
        
        probabilities = torch.matmul(attention, support_y.float() )
        
        return probabilities
     
    
    
    def prediction(self, data_manager_obj, query_x ):
        
        support_x , support_y = data_manager_obj.get_fifo_data( )
        return self.classify_images(query_x, support_x, support_y )
    
    
    '''
    def backward_prediction(self, data_manager_obj, query_x):
        
        predictions = []
        
        for i in range(data_manager_obj.num_old_task_window):
            
            X, Y = data_manager_obj.buffer_x[i], data_manager_obj.buffer_y[i]
            
            unique_labels = torch.unique(Y)
            
            if len(unique_labels) != data_manager_obj.classes_per_task:
                 continue
            
            support_x , support_y = data_manager_obj.get_balaced_task_data( X, Y, unique_labels)
            
            predictions.append( self.classify_images(query_x, support_x, support_y ) )
            
        predictions =  torch.cat(predictions, dim =1)
        predictions  = predictions.argmax(dim = 1)
        predictions = predictions % data_manager_obj.classes_per_task
        
        return predictions
    '''
    
    
    
    
    
    
    