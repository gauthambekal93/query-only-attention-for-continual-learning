# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:56:39 2025

@author: gauthambekal93
"""

import os
import sys
sys.path.append("C:/Users/gauthambekal93/Research/continual_learning/loss_of_plasticity_and_forgetting/common/codes")
sys.path.append("C:/Users/gauthambekal93/Research/continual_learning/loss_of_plasticity_and_forgetting/algorithms/maml/Code")
from pathlib import Path
import json
import torch
import pickle
import argparse
import numpy as np
import random
from tqdm import tqdm
#from bp import Backprop
#from cbp import ContinualBackprop
#from linear import MyLinear
#from torch.nn.functional import softmax
#from deep_ffnn import DeepFFNN
#from miscellaneous import nll_accuracy, compute_matrix_rank_summaries

import torch.nn.functional as F
from collections import deque
import copy
import time
#torch.manual_seed(20)
#np.random.seed(20)
#random.seed(20)




def expr(model_params , data_params):
    agent_type = model_params['agent']
    num_tasks = 200
    if 'num_tasks' in data_params.keys():
        num_tasks = data_params['num_tasks']
    if 'num_examples' in model_params.keys() and "change_after" in model_params.keys():
        num_tasks = int(model_params["num_examples"]/model_params["change_after"])

    step_size = model_params['step_size']
    opt = model_params['opt']
    weight_decay = 0
    use_gpu = 0
    #dev = 'cpu'
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    to_log = False
    num_features = 2000
    change_after = 10 * 6000
    to_perturb = False
    perturb_scale = 0.1
    num_hidden_layers = 1
    mini_batch_size = 400 #10
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'adaptable_contribution'
    if 'to_log' in model_params.keys():
        to_log = model_params['to_log']
    if 'weight_decay' in model_params.keys():
        weight_decay = model_params['weight_decay']
    if 'num_features' in model_params.keys():
        num_features = model_params['num_features']
    if 'change_after' in model_params.keys():
        change_after = model_params['change_after']
    if 'use_gpu' in model_params.keys():
        if model_params['use_gpu'] == 1:
            use_gpu = 1
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if dev == torch.device("cuda"):    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'to_perturb' in model_params.keys():
        to_perturb = model_params['to_perturb']
    if 'perturb_scale' in model_params.keys():
        perturb_scale = model_params['perturb_scale']
    if 'num_hidden_layers' in model_params.keys():
        num_hidden_layers = model_params['num_hidden_layers']
    if 'mini_batch_size' in model_params.keys():
        mini_batch_size = model_params['mini_batch_size']
    if 'replacement_rate' in model_params.keys():
        replacement_rate = model_params['replacement_rate']
    if 'decay_rate' in model_params.keys():
        decay_rate = model_params['decay_rate']
    if 'maturity_threshold' in model_params.keys():
        maturity_threshold = model_params['mt']
    if 'util_type' in model_params.keys():
        util_type = model_params['util_type']

    classes_per_task = 10
    images_per_class = 6000
    input_size = 49 #784
    num_hidden_layers = num_hidden_layers

    #accuracy = nll_accuracy
    examples_per_task = images_per_class * classes_per_task
    total_examples = int(num_tasks * change_after)
    total_iters = int(total_examples/mini_batch_size)
    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks/10)


    iter , test_iter = 0, 0
    
    with open(os.path.join(project_root, data_params['data_dir'] ), 'rb+') as f:
        x, y, _, _ = pickle.load(f)
        if use_gpu == 0:
            x = x.to(dev)
            y = y.to(dev)
            
    y_one_hot = F.one_hot(y, num_classes=classes_per_task).float()  
    y = y.unsqueeze(dim = 1)
    
    train_end_point,  past_task_offset = 58000, 5

    prev_tasks_x , prev_tasks_y_one_hot = deque(maxlen = past_task_offset), deque(maxlen = past_task_offset)
    
    backward_task_samples = 1
    
    num_support, num_query , buffer_tasks= 10, 10, 5 #was 10, 10, 50 
    
    buffer_depth = 60000 #(was 1200
    
    mini_batch_size = 1
    
    buffer = Replay_Buffer(buffer_size = buffer_depth, delete_size= (num_support + num_query)*2, device = dev )
    
    
    maml = MAML(inner_step_size= 0.01, outer_step_size = step_size, input_size= x.shape[1] , num_features=100, num_outputs=1, num_hidden_layers=2, 
                          act_type='relu', opt = opt, step_size = step_size , weight_decay=0,  momentum=0, 
                          beta_1=0.9, beta_2=0.999, loss='nll', task_datapoints = 10000, classes_per_task = classes_per_task,
                          no_of_task_sampled = 5).to(dev)
     
    
    train_accuracies = torch.zeros(total_iters, dtype=torch.float)
    
    backward_accuracies =  torch.zeros( num_tasks, dtype=torch.float)  
    
    forward_accuracies =  torch.zeros( num_tasks , dtype=torch.float)
        
    overall_accuracies =  torch.zeros( num_tasks , dtype=torch.float)
    
    forward_effective_ranks = torch.zeros( num_tasks , dtype=torch.float)
    
    backward_effective_ranks = torch.zeros( num_tasks , dtype=torch.float)
    
    for task_idx in (range(num_tasks)):
        
        print("Task Index ", task_idx)
                
        new_iter_start = iter 
        
        pixel_permutation = np.random.permutation( x.shape[1] )
        
        x = x[:, pixel_permutation]
        
        data_permutation = np.random.permutation(examples_per_task)
        
        x, y, y_one_hot = x[data_permutation], y[data_permutation], y_one_hot[data_permutation]
        
        for start_idx in tqdm(range(0, change_after, mini_batch_size)):
            
            start_idx = start_idx % examples_per_task
          
            """Split data between train and test """
                
            if ( start_idx < ( train_end_point ) or ( task_idx < past_task_offset ) ) :
                
                if  random.random() >0.90:
                    
                    buffer.add_new_data( x[start_idx : start_idx + mini_batch_size], 
                                    y[start_idx : start_idx + mini_batch_size], 
                                    y_one_hot [start_idx : start_idx + mini_batch_size] )
               
        
                    if ( len(buffer.dataset) >= buffer_depth ): # and ( 1 == random.randint(1, 2000) ):
                        
                        
                        support_dict, query_dict = buffer.sample_task_data( buffer_tasks, num_support, num_query )
                        
                        train_loss, train_accuracy = maml.learn(support_dict, query_dict) 
                        
                        train_accuracies[iter] = train_accuracy
                     
                        iter += 1   
                        
            else:
                
                """If we are just begining new task then we need data from old task for backward evaluvation.
                   This is not applicable for forward task since, when you begin a new task you have all the data for new task."""
                
                """Test on previous task """   
                
                for t in range( backward_task_samples ):
                  
                    support_x_backward, support_y_backward = prev_tasks_x[t][train_end_point - num_support : train_end_point] , prev_tasks_y_one_hot[t][train_end_point - num_support : train_end_point]
                    
                    queries_x_backward, queries_y_backward = prev_tasks_x[t][train_end_point:], prev_tasks_y_one_hot[t][train_end_point:]
                    
                    _, backward_accuracy = maml.test( support_x_backward, support_y_backward, queries_x_backward, queries_y_backward ) 
                    
                    backward_accuracies[test_iter] = backward_accuracies[test_iter] + backward_accuracy
                    
                    overall_accuracies[test_iter] = overall_accuracies[test_iter] + backward_accuracy
   
                
                backward_accuracies[test_iter] = backward_accuracies[test_iter] / backward_task_samples
                
                
                """Test on current task """
                support_x_forward, support_y_forward = x[train_end_point - num_support : train_end_point], y_one_hot[train_end_point - num_support : train_end_point]
                
                queries_x_forward, queries_y_forward = x[train_end_point: ], y_one_hot[train_end_point: ]
                
                _, forward_accuracy = maml.test( support_x_forward, support_y_forward, queries_x_forward, queries_y_forward ) 
        
                forward_accuracies[test_iter] =  forward_accuracy
                
                overall_accuracies[test_iter] =  overall_accuracies[test_iter] + forward_accuracy
                
                overall_accuracies[test_iter] =  overall_accuracies[test_iter] / ( backward_task_samples + 1 )
            
                if ( task_idx % 5 == 0) or ( task_idx == past_task_offset):
                    
                    forward_effective_ranks[test_iter] =  maml.calculate_hessian( support_x_forward, support_y_forward, queries_x_forward, queries_y_forward)
                  
                    backward_effective_ranks[test_iter] = maml.calculate_hessian(support_x_backward, support_y_backward, queries_x_backward, queries_y_backward)
                else:
                    forward_effective_ranks[test_iter] = forward_effective_ranks[test_iter -1]
                    
                    backward_effective_ranks[test_iter] = backward_effective_ranks[test_iter - 1]
                
                
                test_iter += 1
                
                break
            
        prev_tasks_x.append(x.clone())
         
        prev_tasks_y_one_hot.append( y_one_hot.clone())
                
        
        print('Train accuracy: ', train_accuracies[new_iter_start:iter - 1].mean().item(), 
       'Backward accuracy: ', backward_accuracies[ test_iter -1 ].item(),
       'Forward accuracy: ',  forward_accuracies[ test_iter -1 ].item(),
       'Overall accuracy: ', overall_accuracies[ test_iter -1 ].item(),
       'forward_effective_ranks: ', forward_effective_ranks[ test_iter -1 ].item(),
       'backward_effective_ranks: ', backward_effective_ranks[ test_iter -1 ].item(),
             )
        
        
        save_after_every_n_tasks = 10
        
        '''
        if task_idx % save_after_every_n_tasks == 0:
            data = {
                   'train_accuracies': train_accuracies.cpu(),
                   'backward_accuracies': backward_accuracies.cpu(),
                   'forward_accuracies': forward_accuracies.cpu(),
                   'overall_accuracies': overall_accuracies.cpu(),
                   'forward_effective_ranks': forward_effective_ranks.cpu(),
                  'backward_effective_ranks': backward_effective_ranks.cpu()

                 }
            result_path = os.path.join( project_root, model_params['model_dir'], 'output.pkl' )
            
            with open(result_path, 'wb+') as f:
                 pickle.dump(data, f)
                 
            model_path = os.path.join(project_root, model_params['model_dir'], 'model.pth' )
           
            torch.save(maml.state_dict(), model_path ) 
        '''
    
def set_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

   
def import_modules():
    
    from replay_buffer_permuted_mnist import Replay_Buffer
    from maml_network_permuted_mnist_2 import MAML
    global Replay_Buffer, MAML



def main(arguments):
   parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
   parser.add_argument('-c1', help="Path to the file containing the parameters for the experiment", type=str)
   parser.add_argument('-c2', help="Path to the file containing the parameters for the experiment", type=str)
   
   args = parser.parse_args(arguments)
  
   with open(args.c1, 'r') as f:
      model_params = json.load(f)
  
   with open(args.c2, 'r') as f:
      data_params = json.load(f)
      
   set_seed(model_params["seed"])
    
   import_modules()
    
   expr(model_params , data_params)



if __name__ == '__main__':

    
    project_root = os.path.abspath( os.path.join(os.getcwd(), "..","..", ".."))
    
    model_config_path = os.path.join(project_root, "runtime_config","models", "permuted_mnist", "maml","203.json") 
    
    data_config_path = os.path.join(project_root, "runtime_config","data", "permuted_mnist", "0.json")
    
    sys.exit( main ( ['-c1', model_config_path, '-c2', data_config_path ] ) )
     
    
