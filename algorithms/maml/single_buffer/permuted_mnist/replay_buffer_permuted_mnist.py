# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 13:23:22 2025

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

import random
import numpy as np

from collections import deque
#torch.manual_seed(20)
#np.random.seed(20)
#random.seed(20)

class Replay_Buffer:
    
    def __init__(self, buffer_size, delete_size, device   ):
        
        self.buffer_size = buffer_size
        
        self.dataset = deque(maxlen = self.buffer_size)
        
        self.delete_size = delete_size
        
        
        
    """add input and output data to the buffer """    
    def add_new_data(self, x, y, y_one_hot):  #was named add_data()
        
        z = torch.cat( [x, y, y_one_hot], dim = 1)
    
        self.dataset.append(z)
    
     
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
       
               support_dict[i] = torch.cat(list(self.dataset)[start_index : start_index + num_support], dim=0)
               
               query_dict[i]  = torch.cat(list(self.dataset)[start_index + num_support : start_index + num_support + num_query], dim=0)
       
           return support_dict, query_dict
       
    
    
    def create_test_data(self, supports_x, supports_y , queries_x, queries_y, num_support ):
       
       X , Y = [], []
       
       for query_x, query_y in zip(queries_x, queries_y):
           
           query_x = query_x.unsqueeze(dim = 0)
           
           X.append( torch.cat( [ supports_x, query_x  ] , dim = 0 ) )
           
           Y.append ( torch.cat( [ supports_y, query_y  ] , dim = 0 ) )
          
       return X, Y
    
    def delete_old_data(self):  #was named delete_data()
      
        self.dataset = self.dataset[self.delete_size:]
        
    '''
    def sample_data(self):
        
        self.label_positions = {}
        
        for label in range(   self.classes_per_task ):
             self.label_positions[label] = torch.where(self.buffer_y[:, -1] == label )[0]
        
        X, Y = [], []
        
        for query_label in range( self.classes_per_task ):
            
            support_x, support_y = [], []
            
            for support_label, v  in self.label_positions.items():
                
                query_index , support_indices = v[0], v[1:] #was -100, -200 (which is incorrect since it completely overfits) 
                
                if query_label == support_label:
                     
                     query_x = self.buffer_x[query_index ].unsqueeze(dim=0)
                     query_y = self.buffer_y_one_hot[ query_index ].unsqueeze(dim=0)
                     
                     support_x.append( self.buffer_x [ support_indices ] )
                     support_y.append(self.buffer_y_one_hot[ support_indices ] )
                     
                else:    
                     support_x.append( self.buffer_x [ support_indices ] )
                     support_y.append(self.buffer_y_one_hot[ support_indices ] )
            
            support_x = torch.cat( support_x , dim = 0)
            support_y = torch.cat( support_y , dim = 0)
            
            query_x = query_x.expand(support_x.shape[0], query_x.shape[1])
            
            X.append( torch.cat([support_x, query_x, support_y], dim = 1) )
            
            Y.append( query_y )
        
        return X, Y    
        
    '''
    '''
    
        
    '''
    '''
    def test_data(self, queries_x, queries_y): #was originally named forward_test()
        
        X, Y = [], []
        
        for query_x, query_y in zip(queries_x , queries_y ) :
            
            query_x =  query_x.unsqueeze(dim=0)
            
            query_y =  query_y.unsqueeze(dim=0)
            
            support_x, support_y = [], []
            
            for support_label, support_indices  in self.label_positions.items():
            
                support_x.append( self.buffer_x [ support_indices ] )
                
                support_y.append(self.buffer_y_one_hot[ support_indices ] )
                
            support_x = torch.cat( support_x , dim = 0)
             
            support_y = torch.cat( support_y , dim = 0)
             
            query_x = query_x.expand(support_x.shape[0], query_x.shape[1])
            
            X.append( torch.cat([support_x, query_x, support_y], dim = 1) )
            
            Y.append( query_y )
        
        return X, Y      
        
    
    '''
    '''
    def inter_task_test(self, buffer_x, buffer_y, buffer_y_one_hot):
        
        label_positions = {}
        
        for label in range(   self.classes_per_task ):
             label_positions[label] = torch.where(buffer_y[:, -1] == label )[0]
        
        X, Y = [], []
        
        for query_label in range( self.classes_per_task ):
            
            support_x, support_y = [], []
            
            for support_label, v  in label_positions.items():
                
                query_index , support_indices = v[0], v[1:] #was -100, -200 (which is incorrect since it completely overfits) 
                
                if query_label == support_label:
                     
                     query_x = buffer_x[query_index ].unsqueeze(dim=0)
                     query_y = buffer_y_one_hot[ query_index ].unsqueeze(dim=0)
                     
                     support_x.append( buffer_x [ support_indices ] )
                     support_y.append(buffer_y_one_hot[ support_indices ] )
                     
                else:    
                     support_x.append( buffer_x [ support_indices ] )
                     support_y.append(buffer_y_one_hot[ support_indices ] )
            
            support_x = torch.cat( support_x , dim = 0)
            support_y = torch.cat( support_y , dim = 0)
            
            query_x = query_x.expand(support_x.shape[0], query_x.shape[1])
            
            X.append( torch.cat([support_x, query_x, support_y], dim = 1) )
            
            Y.append( query_y )
        
        return X, Y    
    
    
    
    '''
    '''
    def test_data_new(self, support_x, support_y, queries_x, queries_y):
        
        X, Y = [], []
        
        for query_x, query_y in zip(queries_x , queries_y ) :
            
            query_x =  query_x.unsqueeze(dim=0)
            
            query_y =  query_y.unsqueeze(dim=0)
            
            query_x = query_x.expand(support_x.shape[0], query_x.shape[1])
            
            X.append( torch.cat([support_x, query_x, support_y], dim = 1) )
            
            Y.append( query_y )
                
        return X, Y        
    '''      
            
            
            