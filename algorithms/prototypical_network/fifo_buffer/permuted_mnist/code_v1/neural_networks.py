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
    def __init__(self, input_size, embedding_features):
        super().__init__()
        self.relu = nn.ReLU()        
        self.embedding_fc1 = nn.Linear(input_size, embedding_features)

        self.embedding_fc2 = nn.Linear(embedding_features , embedding_features)

        # Initialization
        nn.init.kaiming_uniform_(self.embedding_fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.embedding_fc1.bias)
        
        nn.init.kaiming_uniform_(self.embedding_fc2.weight, nonlinearity='linear')
        nn.init.zeros_(self.embedding_fc2.bias)




    def classify_images(self, query_x, support_x, support_y):
            
            query_embedding = self.relu(self.embedding_fc1(query_x))
            
            query_embedding = self.embedding_fc2(query_embedding)
            
            support_embedding = self.relu(self.embedding_fc1(support_x))
            
            support_embedding = self.embedding_fc2(support_embedding)
            
            
            #ELEMENT WISE ADDITION OF SUPPORT VECTORS FOR RELATION NETWORK
            support_embedding = torch.matmul(support_embedding.T, support_y.float()).T
            
            class_counts = support_y.sum(dim=0)
            
            support_embedding = support_embedding / class_counts.clamp_min(1.0).unsqueeze(1)
            
            
            query_embed_shape = query_embedding.shape
            
            support_embed_shape = support_embedding.shape
            
            query_embedding = query_embedding.unsqueeze(1)
            
            query_embedding = query_embedding.expand(query_embed_shape[0], support_embed_shape[0],  query_embed_shape[1] )
            
            support_embedding = support_embedding.unsqueeze(0)
            
            support_embedding = support_embedding.expand(query_embed_shape[0], support_embed_shape[0],  query_embed_shape[1]  )
            
            differences = (query_embedding - support_embedding )
            
            squared_distances = differences.pow(2).sum(dim=-1)
            
            logits = -squared_distances
            
            return logits
    

    
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
    
    
    
    
    
    
    