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

#torch.manual_seed(20)
#np.random.seed(20)
#random.seed(20)

class Replay_Buffer_OLD:
    
    def __init__(self, feature_size, device, queue_size ):
        
        self.z = torch.zeros(feature_size, queue_size ).to(device)
        
        self.previous_value = None
        
    def create_matrix(self, x, y, zero_tensor):
        
        current_feature = torch.cat((x,  zero_tensor ), dim = 1).reshape(-1, 1)
        
        if self.previous_value is not None:
            self.z[:, -1:] = self.previous_value
            
        temp_z = self.z[:, 1:]
        
        self.z = torch.cat((temp_z, current_feature), dim = 1)
        
        self.previous_value = torch.cat((x, y), dim = 1).reshape(-1, 1)
        
        return self.z
    
    
class Replay_Buffer:
    
    def __init__(self, device):
        self.zero_tensor =  torch.zeros((1,1)).to(device)
        
    def create_train_data(self, x, y ):
        
        z = torch.cat ( [x, y ], dim = 1)
        
        z[-1, -1] = self.zero_tensor 
        
        y_actual = y[-1:, -1:]
        
        return z, y_actual
   
    
    def create_test_data(self, supports_x, supports_y , queries_x, queries_y, num_support ):
       
       z = []
       
       queries = torch.cat( ( queries_x, self.zero_tensor.expand( len(queries_x) , -1) ), dim = 1)
       
       supports = torch.cat( (supports_x, supports_y ), dim = 1)
       
       for query in queries:
           
           query = query.reshape(1, -1)
           
           z.append( torch.cat( (supports, query) , dim = 0 ) )
       
       y_actual = queries_y.clone()
       
       return z, y_actual


  
            