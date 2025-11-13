# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:22:56 2025

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:46:00 2025

@author: gauthambekal93
"""

import time
import torch
#import torch.nn as nn
#from collections import deque

import random
import numpy as np

#torch.manual_seed(20)
#np.random.seed(20)
#random.seed(20)

class Replay_Buffer:
    
    def __init__(self, buffer_depth, classes_per_task, support_size ,queries_per_mini_batch,  device   ):
        
        self.buffer_x,  self.buffer_y , self.buffer_y_one_hot = None, None, None
        
        self.buffer_depth = buffer_depth
        
        self.classes_per_task = classes_per_task
        
        self.support_size = support_size
        
        self.queries_per_mini_batch = queries_per_mini_batch
        
        self.zero_tensor =  torch.zeros((1,self.classes_per_task)).to(device)
        
        
    """add input and output data to the buffer """    
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
        
        data_dict = {}
        
        data_dict["support_x"] = self.buffer_x[:-1]
        
        data_dict["support_y"] = self.buffer_y_one_hot[:-1]
        
        data_dict["query_x"] = self.buffer_x[-1:]
        
        data_dict["query_y"] = self.buffer_y_one_hot[-1:] 
        
        data_dict["zero_padding"] = self.zero_tensor
        
        return data_dict
    
    
    def delete_old_data(self): 
      
        self.buffer_x = self.buffer_x[ - self.buffer_depth: ]
        
        self.buffer_y = self.buffer_y[- self.buffer_depth: ]
        
        self.buffer_y_one_hot = self.buffer_y_one_hot[ - self.buffer_depth: ]
    
    
    
    def test_data_new(self, support_x, support_y, queries_x, queries_y):
        
        X, Y = [], []
        
        for query_x, query_y in zip(queries_x , queries_y ) :
            
            query_x =  query_x.unsqueeze(dim=0)
            
            query_y =  query_y.unsqueeze(dim=0)
            
            query_x = query_x.expand(support_x.shape[0], query_x.shape[1])
            
            X.append( torch.cat([support_x, query_x, support_y], dim = 1) )
            
            Y.append( query_y )
        
          
        return X, Y        
                
