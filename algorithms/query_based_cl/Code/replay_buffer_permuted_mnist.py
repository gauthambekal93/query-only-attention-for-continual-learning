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
    
    def __init__(self, buffer_depth, buffer_width, classes_per_task, support_size ,samples_per_batch,  device   ):
        
        self.buffer_x,  self.buffer_y , self.buffer_y_one_hot = None, None, None
        
        self.buffer_depth = buffer_depth
        
        self.buffer_width = buffer_width
        
        self.classes_per_task = classes_per_task
        
        self.support_size = support_size
        
        self.samples_per_batch = samples_per_batch
        
        
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
       

       self.label_positions = {}
       
       start_idx = torch.randint(0, len(self.buffer_x) -self.support_size, (1,)) 
       
       end_idx = start_idx + self.support_size
       
       #start_idx = 0
       
       #end_idx = start_idx + self.support_size
       
       buffer_x = self.buffer_x[start_idx: end_idx]
       
       buffer_y = self.buffer_y[start_idx: end_idx]
       
       buffer_y_one_hot = self.buffer_y_one_hot[start_idx: end_idx]
       
       for label in range(   self.classes_per_task ):
           
           self.label_positions[label] = torch.where(buffer_y[:, -1] == label )[0]
       
       X, Y = [], []
       
       shuffled_class_labels = torch.randperm(self.classes_per_task).tolist()[: self.samples_per_batch]

       for query_label in shuffled_class_labels :   # range(self.classes_per_task)
           
           support_x, support_y = [], []
           
           for support_label, v  in self.label_positions.items():
               
               query_index , support_indices = v[0], v[1:] 
               
               if query_label == support_label:
                    
                    query_x = buffer_x[query_index ].unsqueeze(dim=0)
                    query_y = buffer_y_one_hot[ query_index ].unsqueeze(dim=0)
                    
                    support_x.append( buffer_x [ support_indices ] )
                    support_y.append( buffer_y_one_hot[ support_indices ] )
                    
               else:    
                    support_x.append( buffer_x [ support_indices ] )
                    support_y.append( buffer_y_one_hot[ support_indices ] )
           
           support_x = torch.cat( support_x , dim = 0)
           support_y = torch.cat( support_y , dim = 0)
           
           query_x = query_x.expand(support_x.shape[0], query_x.shape[1])
           
           X.append( torch.cat([support_x, query_x, support_y], dim = 1) )
           
           Y.append( query_y )
       
       return X, Y    
    
    
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

            
            
    
    
    