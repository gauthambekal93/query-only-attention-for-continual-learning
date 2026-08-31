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
    def __init__(self, input_size = 49, embedding_features = 40 , relational_features = 37):
        super().__init__()
        self.relu = nn.ReLU()        
        self.embedding_fc = nn.Linear(input_size, embedding_features)

        self.fc1 = nn.Linear(embedding_features * 2, relational_features)
        self.fc2 = nn.Linear(relational_features, relational_features)
        self.fc3 = nn.Linear(relational_features, relational_features)
        self.fc4 = nn.Linear(relational_features, relational_features)
        self.fc5 = nn.Linear(relational_features, 1)
          
        # Initialization
        nn.init.kaiming_uniform_(self.embedding_fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.embedding_fc.bias)
        
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
        
        query_embedding = self.relu(self.embedding_fc(query_x))
        
        support_embedding = self.relu(self.embedding_fc(support_x))

        #ELEMENT WISE ADDITION OF SUPPORT VECTORS FOR RELATION NETWORK
        support_embedding = torch.matmul(support_embedding.T, support_y.float()).T
        
        query_embed_shape = query_embedding.shape
        
        support_embed_shape = support_embedding.shape
        
        query_embedding = query_embedding.unsqueeze(1)
        
        query_embedding = query_embedding.expand(query_embed_shape[0], support_embed_shape[0],  query_embed_shape[1] )
        
        support_embedding = support_embedding.unsqueeze(0)
        
        support_embedding = support_embedding.expand(query_embed_shape[0], support_embed_shape[0],  query_embed_shape[1]  )
        
        x = torch.cat([query_embedding, support_embedding], dim=-1)
        
        x = self.relu ( self.fc1(x) )     
        
        x = self.relu ( self.fc2(x) )               
        
        x = self.relu ( self.fc3(x) )  
        x = self.relu ( self.fc4(x) )  
        x = torch.sigmoid( self.fc5(x))
        x = x.reshape(-1, x.shape[1])
        return x
    

        
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
    
    
    
    
    
    
    