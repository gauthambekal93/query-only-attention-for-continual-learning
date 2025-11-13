# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:46:00 2025

@author: gauthambekal93
"""


import torch
#import torch.nn as nn
#from collections import deque

import random
import numpy as np

#torch.manual_seed(20)
#np.random.seed(20)
#random.seed(20)
from collections import deque


class Replay_Buffer:
    
    def __init__(self, buffer_size):
        
        self.buffer_size = buffer_size
        
        self.dataset = deque(maxlen = self.buffer_size)
    
    def add_data(self, x, y):
        
        z = torch.cat( [x, y], dim = 0)
        
        self.dataset.append(z)
    
    def sample_task_data_old(self, num_tasks, num_support, num_query):
        
        support_dict, query_dict = {}, {}
        
        seed_num = random.randint(0, len(self.dataset) - (num_support + num_query) )
        
        distance = int(len(self.dataset) / num_tasks)
        
        task_start_indices = [ abs ( seed_num - distance * i ) for i in range(num_tasks)]
        
        for i, start_index in enumerate( task_start_indices):
            try:
                support_dict[i] =  torch.stack( list(self.dataset)[start_index : start_index + num_support], dim = 0 ) 
                
                query_dict[i] = torch.stack( list(self.dataset)[start_index + num_support : start_index + num_support + num_query ], dim = 0 ) 
            except: 
                    print("stop")
                    print("stop")
               
        return support_dict, query_dict
    


    def sample_task_data(self, num_tasks, num_support, num_query):
        
        support_dict, query_dict = {}, {}
        
        total = len(self.dataset)
        
        step = total // num_tasks  # e.g., 2501 if total = 10006
    
        for i in range(num_tasks):
            # define region for task i
            region_start = i * step
            
            region_end = region_start + step
    
            # make sure we have enough room
            max_start = region_end - (num_support + num_query)
            
            if max_start <= region_start:
                raise ValueError("Region too small for support + query")
    
            # randomly choose a start point in this region
            start_index = random.randint(region_start, max_start)
    
            support_dict[i] = torch.stack(list(self.dataset)[start_index : start_index + num_support], dim=0)
            
            query_dict[i]  = torch.stack(list(self.dataset)[start_index + num_support : start_index + num_support + num_query], dim=0)
    
        return support_dict, query_dict
    
    '''
    def create_train_data(self, x, y, num_support ):
        
        supports_x =  x[:-1] 
        
        supports_y =  y[:-1] 
        
        query_x =   x[-1:] 
        
        query_y =   y[-1:] 
        
        X = torch.cat( [ supports_x, query_x  ] , dim = 0 )
        
        Y = torch.cat( [ supports_y, query_y  ] , dim = 0 )
       
        return X, Y
   
   '''
   
    def create_test_data(self, supports_x, supports_y , queries_x, queries_y, num_support ):
       
       X , Y = [], []
       
       for query_x, query_y in zip(queries_x, queries_y):
           
           query_x = query_x.unsqueeze(dim = 0)
           
           #query_x = query_x.expand( num_support, -1)
           
           X.append( torch.cat( [ supports_x, query_x  ] , dim = 0 ) )
           
           Y.append ( torch.cat( [ supports_y, query_y  ] , dim = 0 ) )
          
       return X, Y


  
        
    
    
    
            
            
            