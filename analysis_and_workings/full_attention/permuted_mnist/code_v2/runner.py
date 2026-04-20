# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:15:04 2025

@author: gauthambekal93
"""


import torch
from tqdm import tqdm
import time
import torch.nn.functional as F
import numpy as np

class Runner:
    
    def __init__(self, num_datapoints_per_timestep, test_batch_size):
        
        self.num_datapoints_per_timestep = num_datapoints_per_timestep
        self.test_batch_size = test_batch_size
        
        
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        checkpoint_obj.load_experiment_checkpoint(train_context, data_manager_obj)
        
        train_context.net.eval()
        
        pair_wise_attentions = []
        
        distance_metric = { i: [] for i in range (0, 10)}
        #distance_metric = []
        
        for task_id in range(1, 1000):
            
            data_manager_obj.create_task_data(task_id)
            
            train_x = data_manager_obj.task_train_x[task_id]
            
            train_y = data_manager_obj.task_train_y[task_id]
            
            batch_x  = train_x[ : self.num_datapoints_per_timestep] 
            
            batch_y = train_y[ : self.num_datapoints_per_timestep]
            
            data_manager_obj.fill_fifo_buffer( batch_x, batch_y )
            
            label_specific_attention = train_context.net.prediction(data_manager_obj, batch_x, batch_y)
            
            pair_wise_attentions.append(label_specific_attention)
            
            
            if len(pair_wise_attentions) ==2 :
        
                for ( label1, att1 ), (label2, att2) in zip(pair_wise_attentions[0].items(), pair_wise_attentions[1].items()):
                    #att1 = torch.tensor( [att1[i:i+10].sum().item() for i in range(0, len(att1), 10)])
                    #att2 = torch.tensor( [att2[i:i+10].sum().item() for i in range(0, len(att2), 10)])
                    
                    att1 = torch.tensor( [att1[i:i+10].mean().item() for i in range(0, len(att1), 10)])
                    att2 = torch.tensor( [att2[i:i+10].mean().item() for i in range(0, len(att2), 10)])
                    
                    distance_metric[label1].append( torch.norm(att1 - att2, p=2 ).item() )
                    #distance_metric[label1].append( F.cosine_similarity(att1, att2, dim=0) .item())
         
                pair_wise_attentions = []
                
                data_manager_obj.delete_data(task_id)
        
        for i in range(10):
            print( "Label ", i, "distance", np.mean(distance_metric[i]))
            
        '''    
        num_pairs = len(distance_metric[0])
        
        task_avg_distance_metric = []
        
        for i in range(num_pairs):
            temp = []
            
            for j in range(10):
                temp.append( distance_metric[j][i] )
            task_avg_distance_metric.append( np.mean(temp) )
             
        print("Distance metric ", task_avg_distance_metric)    
        '''
        