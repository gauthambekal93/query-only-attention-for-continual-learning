# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 18:29:43 2025

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 11 08:48:36 2025

@author: gauthambekal93
"""

import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
import sys
sys.path.append(str(BASE_DIR / "common" / "codes"))
sys.path.append ( str(BASE_DIR / "algorithms" / "transformer"/ "Code") )

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
    dev = 'cpu'
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
        if use_gpu == 1:
            x = x.to(dev)
            y = y.to(dev)
            
    y_one_hot = F.one_hot(y, num_classes=classes_per_task).float()  
    y = y.unsqueeze(dim = 1)
    

    train_end_point,  past_task_offset = 58000, 5  

    prev_tasks_x , prev_tasks_y_one_hot = deque(maxlen = past_task_offset), deque(maxlen = past_task_offset)
    
    backward_task_samples = 1
    
    buffer_depth = 100 #1200  took 1 hr per task!, 100 needs to be tried again.
    
    support_size = 100
    
    buffer = Replay_Buffer(buffer_depth, mini_batch_size, classes_per_task, dev )
    
    """ STEP SIZE IS SET TO 0.0005 TO MATCH WITH THE PAPER PARAMETERS. MAY NEED TO CHANGE LATER!!!"""
    transformer = Transformers(feature_size = x.shape[1] + y_one_hot.shape[1] , num_outputs=10, num_hidden_layers=2, 
                          act_type='relu', opt = opt, step_size = 0.0005 , weight_decay=0,  momentum=0, 
                          beta_1=0.9, beta_2=0.999, loss='nll', task_datapoints = 10000, classes_per_task = classes_per_task).to(dev)
    
    train_losses = torch.zeros(total_iters, dtype=torch.float)
    
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
        
        """This should be uncommented later on """
        x = x[:, pixel_permutation]    
        
        data_permutation = np.random.permutation(examples_per_task)
        
        x, y, y_one_hot = x[data_permutation], y[data_permutation], y_one_hot[data_permutation]
        
       
        for start_idx in tqdm(range(0, change_after, mini_batch_size)):
            
            start_idx = start_idx % examples_per_task
            
            """Split data between train and test """
            
            if start_idx < ( train_end_point ) or ( task_idx < past_task_offset ):
                
                buffer.add_new_data( x [start_idx : start_idx + mini_batch_size], 
                                     y [start_idx : start_idx + mini_batch_size], 
                                     y_one_hot [start_idx: start_idx + mini_batch_size] )
                
                buffer.delete_old_data()
                
                if len(buffer.buffer_x)>= buffer_depth:
                    
                    batch_x, batch_y = buffer.sample_data()
                    
                    train_loss, train_accuracy = transformer.learn(batch_x, batch_y) 
                
                    train_accuracies[iter] = train_accuracy
                    
                    train_losses[iter] = train_loss
                    
                    iter += 1   
            
            else:
                
                with torch.no_grad():
                    
                
                    for t in range( backward_task_samples ):
                      
                        support_x, support_y = prev_tasks_x[t][:support_size] , prev_tasks_y_one_hot[t][:support_size]
                        
                        queries_x, queries_y = prev_tasks_x[t][train_end_point:], prev_tasks_y_one_hot[t][train_end_point:]
                        
                        test_x_backward, test_y_backward = buffer.create_test_data( support_x, support_y , queries_x, queries_y )
                        
                        _, backward_accuracy = transformer.test( test_x_backward, test_y_backward ) 
                        
                        backward_accuracies[test_iter] = backward_accuracies[test_iter] + backward_accuracy
                        
                        overall_accuracies[test_iter] = overall_accuracies[test_iter] + backward_accuracy
                        
                        break
                    
                    
                    backward_accuracies[test_iter] = backward_accuracies[test_iter] / backward_task_samples
                    
                    
                    """Test on current task """
                    support_x, support_y = x[:support_size], y_one_hot[: support_size]
                    
                    queries_x, queries_y = x[train_end_point: ], y_one_hot[train_end_point: ]
                    
                    test_x_forward, test_y_forward = buffer.create_test_data( support_x, support_y , queries_x, queries_y )
                    
                    _, forward_accuracy = transformer.test( test_x_forward, test_y_forward ) 
            
                    forward_accuracies[test_iter] =  forward_accuracy
                    
                    overall_accuracies[test_iter] =  overall_accuracies[test_iter] + forward_accuracy
                    
                    overall_accuracies[test_iter] =  overall_accuracies[test_iter] / ( backward_task_samples + 1 )
                    
                '''
                if task_idx % 100 ==0:
                    
                    forward_effective_ranks[test_iter] =  transformer.calculate_hessian( test_z_forward, test_y_forward)
                  
                    backward_effective_ranks[test_iter] = transformer.calculate_hessian(test_z_backward, test_y_backward)
                else:
                    forward_effective_ranks[test_iter] = forward_effective_ranks[test_iter -1]
                    
                    backward_effective_ranks[test_iter] = backward_effective_ranks[test_iter - 1]
                '''    
                    
                    
                if (task_idx % 100 ==0) or ( task_idx == past_task_offset):
                 
                    start_idx = random.randrange(0, len(test_x_forward))
                    
                    end_idx = start_idx + 100
                    
                    forward_effective_ranks[test_iter] =  transformer.calculate_hessian( test_x_forward[start_idx: end_idx], test_y_forward[start_idx: end_idx])
                      
                    backward_effective_ranks[test_iter] = transformer.calculate_hessian(test_x_backward[start_idx: end_idx], test_y_backward[start_idx: end_idx])
                
                else:
                 
                    forward_effective_ranks[test_iter] = forward_effective_ranks[test_iter -1]
                    
                    backward_effective_ranks[test_iter] = backward_effective_ranks[test_iter - 1]
                 
                 
                test_iter += 1
                
                break
                
            
        prev_tasks_x.append(x.clone())
         
        prev_tasks_y_one_hot.append( y_one_hot.clone())
                
        
        print('Train accuracy: ', train_accuracies[new_iter_start:iter - 1].mean().item(),
              'Train Loss: ',    train_losses[new_iter_start:iter - 1].mean().item(),
              'Backward accuracy: ', backward_accuracies[ test_iter -1 ].item(),
              'Forward accuracy: ',  forward_accuracies[ test_iter -1 ].item(),
              'Overall accuracy: ', overall_accuracies[ test_iter -1 ].item(),
              'forward_effective_ranks: ', forward_effective_ranks[ test_iter -1 ].item(),
              'backward_effective_ranks: ', backward_effective_ranks[ test_iter -1 ].item(),
             )
        
        '''
        if task_idx % save_after_every_n_tasks == 0:
            data = {
                   'train_accuracies': train_accuracies.cpu(),
                   'train_loss': train_losses.cpu(),
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
           
            torch.save(transformer.state_dict(), model_path ) 
        '''         
                
                
  


def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

   
def import_modules():
    global  Replay_Buffer, Transformers
    
    from replay_buffer_permuted_mnist import Replay_Buffer
    from transformer_permuted_mnist import Transformers
    
    
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
    
    model_config_path = os.path.join(project_root, "runtime_config","models", "permuted_mnist", "transformer","202.json")
    
    data_config_path = os.path.join(project_root, "runtime_config","data", "permuted_mnist", "0.json")
    
    main( ['-c1', model_config_path, '-c2', data_config_path ] )
    
