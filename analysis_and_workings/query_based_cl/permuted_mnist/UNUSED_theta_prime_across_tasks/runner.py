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
        
    
    def prequential_testing(self, train_context, data_manager_obj, batch_x, batch_y):
        
        train_context.net.eval()
        
        with torch.no_grad():
            predictions = train_context.net.prediction(data_manager_obj, batch_x)
        
        accuracy = 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
        
        return accuracy
                
    
    def forward_testing(self, train_context, data_manager_obj):
        
        train_context.net.eval()
        
        batch_x, batch_y = data_manager_obj.task_test_x[data_manager_obj.current_task_id], data_manager_obj.task_test_y[ data_manager_obj.current_task_id ]
        
        with torch.no_grad():
            predictions = train_context.net.prediction(data_manager_obj, batch_x)
        
        accuracy = 100 * torch.mean((predictions.argmax(axis=1) == batch_y).to(torch.float32)).item()
        
        return accuracy
        
     
 
    def backward_testing(self, train_context, data_manager_obj):
         
         return 0
    
         train_context.net.eval()
         
         avg_acc = 0.0
         sub_task_accuracies = {}

         with torch.no_grad():
             
             for task_id in data_manager_obj.task_test_x.keys():
                 
                 if task_id != data_manager_obj.current_task_id:
                 
                     batch_x, batch_y = data_manager_obj.task_test_x[ task_id ], data_manager_obj.task_test_y[ task_id ]
                      
                     predictions = train_context.net.prediction( data_manager_obj,  batch_x)
                     
                     accuracy = 100 * torch.mean((predictions == batch_y).to(torch.float32)).item()
                     
                     avg_acc += accuracy
                     
                     sub_task_accuracies[task_id] = accuracy
  
         accuracy =  avg_acc / ( len(data_manager_obj.task_test_x.keys() ) - 1 ) 
         
         return accuracy
         
    
        
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        checkpoint_obj.load_experiment_checkpoint(train_context, data_manager_obj)
        
        train_context.net.eval()
        
        pair_wise_attentions = []
        
        distance_metric = { i: [] for i in range (0, 10)}
        #distance_metric = []
        
        for task_id in range(1, 100):
            
            data_manager_obj.create_task_data(task_id)
            
            train_x = data_manager_obj.task_train_x[task_id]
            
            train_y = data_manager_obj.task_train_y[task_id]
            
            batch_x  = train_x[ : self.num_datapoints_per_timestep] 
            
            batch_y = train_y[ : self.num_datapoints_per_timestep]
            
            query_x, query_y = batch_x[:100] , batch_y[:100]
            
            support_x, support_y = batch_x[100:] , batch_y[100:]
            
            data_manager_obj.fill_fifo_buffer( support_x, support_y  )
            
            label_specific_generated_params = train_context.net.prediction( data_manager_obj, query_x , query_y)
            
            pair_wise_attentions.append(label_specific_generated_params)
                       
            if len(pair_wise_attentions) ==2 :
        
                for ( label1, att1 ), (label2, att2) in zip(pair_wise_attentions[0].items(), pair_wise_attentions[1].items()):
                    
                    att1 = torch.tensor( [att1[i:i+10].mean()for i in range(0, len(att1), 10)])
                    att1 = F.softmax(att1, dim=0)
                    
                    att2 = torch.tensor( [att2[i:i+10].mean() for i in range(0, len(att2), 10)])
                    att2 = F.softmax(att2, dim=0)
                    
                    distance_metric[label1].append( torch.norm(att1 - att2, p=2 ).item() )
                
           
                #distance_metric.append( torch.norm(pair_wise_attentions[0] - pair_wise_attentions[1], p=2 ).item() )
                     
                pair_wise_attentions = []
                
                data_manager_obj.delete_data(task_id)
                
        for i in range(10):
            print( "Label ", i, "distance", np.mean(distance_metric[i]))
            
        
        '''
        while data_manager_obj.current_task_id < data_manager_obj.num_tasks: 
            
            start = time.perf_counter()
        
            data_manager_obj.create_task_data()

            if  ( data_manager_obj.current_task_id >= data_manager_obj.num_old_task_window ) :
                
                train_loss, train_accuracy, prequential_accuracy, forward_accuracy, backward_accuracy = self.train( train_context, data_manager_obj, checkpoint_obj)
                
                checkpoint_obj.save_model_checkpoint( train_context, data_manager_obj, train_loss, data_manager_obj.current_task_id)
                
                checkpoint_obj.save_result_checkpoint(data_manager_obj, train_loss, train_accuracy, prequential_accuracy, forward_accuracy, backward_accuracy)
                
            if data_manager_obj.current_task_id >= data_manager_obj.num_old_task_window: 
                
                data_manager_obj.delete_data()
                
                 
            data_manager_obj.current_task_id += 1
            
            print("Loop time ", time.perf_counter() -  start)
            
            print("===========================================================================================")
            
        '''   

