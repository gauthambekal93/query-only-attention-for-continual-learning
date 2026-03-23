# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:46:00 2025

@author: gauthambekal93
"""


import torch

class Replay_Buffer:
    
    def create_train_data(self, x, y, num_support ):
        
        z = torch.cat ( [x, y ], dim = 1)
        
        supports_x =  z[:num_support, :-1 ]
        
        supports_y =  z[:num_support, -1: ]
    
        query_x =   z[num_support:, : -1 ]
        
        query_x = query_x.expand( num_support,-1)
        
        X = torch.cat( [ supports_x, query_x , supports_y ] , dim = 1 )
        
        Y =   z[num_support:, -1: ]
    
        return X, Y
   
    
    def create_test_data(self, supports_x, supports_y , queries_x, queries_y, num_support ):
       
       X , Y = [], []
       
       for query_x, query_y in zip(queries_x, queries_y):
           
           query_x = query_x.unsqueeze(dim = 0)
           
           query_x = query_x.expand( num_support, -1)
           
           X.append( torch.cat( [ supports_x, query_x , supports_y ] , dim = 1 ) )
           
           Y.append(query_y)
   
       return X, Y


  
        
    
    
    
            
            
            