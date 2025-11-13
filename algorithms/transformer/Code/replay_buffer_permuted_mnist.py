# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:46:00 2025

@author: gauthambekal93
"""


import torch
import torch.nn as nn
from collections import deque

import random
import numpy as np


    
    
class Replay_Buffer:
    
    def __init__(self, buffer_depth, batch_size, classes_per_task, device):
        
        self.batch_size = batch_size
        
        self.z = None
        
        self.buffer_depth = buffer_depth
        
        self.classes_per_task = classes_per_task
        
        self.zero_tensor =  torch.zeros((1,self.classes_per_task)).to(device)
        
        self.buffer_x, self.buffer_y, self.buffer_y_one_hot = None, None, None
        
    """add data to the buffer """    
    def add_new_data(self, x, y, y_one_hot):  #was named add_data()
        
        if self.buffer_x is None:
            self.buffer_x = x
            self.buffer_y = y
            self.buffer_y_one_hot = y_one_hot
        else:
            self.buffer_x = torch.cat ( [self.buffer_x, x ], dim = 0)
            
            self.buffer_y = torch.cat ( [self.buffer_y, y ], dim = 0)
            
            self.buffer_y_one_hot = torch.cat ( [self.buffer_y_one_hot, y_one_hot ], dim = 0)  
            
            
    def sample_data(self):
        
        support = torch.cat( (self.buffer_x[:-1], self.buffer_y_one_hot[:-1]) , dim = 1 )
        
        query = torch.cat( (self.buffer_x[-1:], self.zero_tensor) , dim = 1 )
        
        X = torch.cat( (support, query), dim = 0)
          
        Y_actual = self.buffer_y_one_hot[-1:].clone()
         
        return X, Y_actual
     
        
    def delete_old_data(self):  
         
         self.buffer_x = self.buffer_x[ - self.buffer_depth: ]
         
         self.buffer_y = self.buffer_y[ - self.buffer_depth: ]
         
         self.buffer_y_one_hot = self.buffer_y_one_hot[ - self.buffer_depth: ]
         
     
        
    def create_test_data(self, supports_x, supports_y , queries_x, queries_y ):
       #we donot want the origibal source of data to be updated when we zero out some tensors for creating the data
       supports_x, supports_y , queries_x, queries_y  =  supports_x.clone(), supports_y.clone() , queries_x.clone(), queries_y.clone()
       
       supports =  torch.cat( (supports_x, supports_y), dim = 1)[1:]
       
       x , y_actual = [], []
       
       for query_x, query_y in zip(queries_x, queries_y):
           
           y_actual.append(query_y.clone())
           
           query_y[:] = self.zero_tensor    #possible error here ???
           
           #query = torch.cat( (query_x, self.zero_tensor.reshape(-1)), dim = 0).reshape(1, -1)
           
           query = torch.cat( (query_x, query_y), dim = 0).reshape(1, -1)
           
           x.append( torch.cat( (supports, query), dim = 0 ) ) 
       
       #z = torch.stack( z, dim = 0)
       
       #y_actual = torch.stack( y_actual, dim = 0)
       
       return x, y_actual


  
            