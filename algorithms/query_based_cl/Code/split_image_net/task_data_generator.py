# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 11:39:04 2025

@author: gauthambekal93
"""

import random
import torch

import torch.nn.functional as F

def generate_train_data(x, label_ids, images_per_class, classes_per_task):
    
    label_1, label_2 = random.sample(label_ids[:-1] , 2)
    
    start_idx, end_idx = label_1 * images_per_class, (label_1 + 1) *images_per_class
    
    x_1, y_1 = x[start_idx: end_idx], torch.zeros(images_per_class, dtype = torch.long)
    
    start_idx, end_idx = (label_2)*images_per_class, (label_2 + 1)*images_per_class
    
    x_2, y_2 = x[start_idx: end_idx], torch.ones(images_per_class, dtype = torch.long)
    
    train_x, train_y =  torch.cat([x_1, x_2], dim = 0), torch.cat([y_1, y_2], dim = 0)
    
    y_one_hot = F.one_hot(train_y, num_classes=classes_per_task).float() 
    
    train_y = train_y.unsqueeze(dim = 1)
    
    perm = torch.randperm(len(train_x))
    
    train_x, train_y, y_one_hot = train_x[perm], train_y[perm], y_one_hot[perm]
    
    return train_x, train_y, y_one_hot, label_1, label_2



def generate_test_data(data_dict, image_net_val_x, image_net_val_y, label_1, label_2 , images_per_class, classes_per_task ):
    
    support_x, support_y =  data_dict["support_x"][-1], data_dict["support_y"][-1]
    
    idx1 = torch.where(image_net_val_y == label_1)[0]  
    
    idx2 = torch.where(image_net_val_y == label_2)[0]  
    
    data_dict_val =  { "query_x":[], "query_y":[],  "support_x":[], "support_y":[] }
    
    queries_x = torch.cat( (image_net_val_x[idx1], image_net_val_x[idx2]), dim =0)
    
    y = torch.cat( ( torch.zeros(images_per_class, dtype = torch.long), torch.ones(images_per_class, dtype = torch.long) ), dim =0)
    
    queries_y = F.one_hot(y, num_classes=classes_per_task).float() 

    for query_x, query_y in zip(queries_x, queries_y):
        
        query_x =  query_x.unsqueeze(dim=0)
        
        query_y =  query_y.unsqueeze(dim=0)
        
        query_x = query_x.expand(len(support_x), -1, -1, -1)
    
        data_dict_val["support_x"].append(support_x)
         
        data_dict_val["support_y"].append(support_y)
        
        data_dict_val["query_x"].append( query_x)
        
        data_dict_val["query_y"].append( query_y)
    
    return data_dict_val