# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 15:44:20 2025

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 17:22:17 2025

@author: gauthambekal93
"""


import os
import sys

from pathlib import Path
# Get current file's directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

# Add it to sys.path
sys.path.append(str(BASE_DIR / "common" / "codes"))


import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import copy

import random
from collections import deque

#torch.manual_seed(20)
#np.random.seed(20)
#random.seed(20)


def expr(model_params , data_params):
    agent_type = model_params['agent']
    num_tasks = 200
    if 'num_tasks' in data_params.keys():
        num_tasks = data_params['num_tasks']
    if 'num_examples' in data_params.keys() and "change_after" in data_params.keys():
        num_tasks = int(data_params["num_examples"]/data_params["change_after"])

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
    mini_batch_size = 400 #1
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
    net = DeepFFNN(input_size=input_size, num_features=num_features, num_outputs=classes_per_task, num_hidden_layers=num_hidden_layers)

    if agent_type == 'linear':
        net = MyLinear(
            input_size=input_size, num_outputs=classes_per_task
        )
        net.layers_to_log = []

    if agent_type in ['bp', 'linear', "l2"]:
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            weight_decay=weight_decay,
            device=dev,
            to_perturb=to_perturb,
            perturb_scale=perturb_scale,
        )
    elif agent_type in ['cbp']:
        learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            decay_rate=decay_rate,
            util_type=util_type,
            accumulate=True,
            device=dev,
        )

    accuracy = nll_accuracy
    examples_per_task = images_per_class * classes_per_task
    total_examples = int(num_tasks * change_after)
    total_iters = int(total_examples/mini_batch_size)
    save_after_every_n_tasks = 1
    
    if num_tasks >= 10:
        #save_after_every_n_tasks = int(num_tasks/10)
        save_after_every_n_tasks = 100
        
    train_end_point,  past_task_offset = 58000, 5 
    
    prev_tasks_x , prev_tasks_y = deque(maxlen = past_task_offset), deque(maxlen = past_task_offset)
    
    backward_task_samples = 1
    
    train_accuracies = torch.zeros(total_iters, dtype=torch.float)
    
    backward_accuracies =  torch.zeros( num_tasks , dtype=torch.float)  
    
    forward_accuracies =  torch.zeros( num_tasks , dtype=torch.float)
    
    overall_accuracies =  torch.zeros( num_tasks , dtype=torch.float)
   
    forward_effective_ranks = torch.zeros( num_tasks , dtype=torch.float)
    
    backward_effective_ranks = torch.zeros( num_tasks , dtype=torch.float)
    
    iter, test_iter = 0, 0
    
    with open(os.path.join(project_root, data_params['data_dir'] ), 'rb+') as f:
        x, y, _, _ = pickle.load(f)
        if use_gpu == 1:
            x = x.to(dev)
            y = y.to(dev)
    

    for task_idx in (range(num_tasks)):
        
        print("Task Index ", task_idx)
        
        new_iter_start = iter 
        
        pixel_permutation = np.random.permutation(input_size)
        x = x[:, pixel_permutation]
        data_permutation = np.random.permutation(examples_per_task)
        x, y = x[data_permutation], y[data_permutation]
         

        for start_idx in tqdm(range(0, change_after, mini_batch_size)):
            
            start_idx = start_idx % examples_per_task
            
            batch_x = x[start_idx: start_idx+mini_batch_size]
            
            batch_y = y[start_idx: start_idx+mini_batch_size]
            
            if start_idx < ( train_end_point ) or ( task_idx < past_task_offset ):
                # train the network
                loss, network_output = learner.learn(x=batch_x, target=batch_y)

                # log accuracy
                with torch.no_grad():
                    train_accuracies[iter] = accuracy(softmax(network_output, dim=1), batch_y).cpu()*100
                
                iter += 1 
            else:
                
                with torch.no_grad():
                    
                    test_batch_size = len(x) - train_end_point
                    
                    """If we are just begining new task then we need data from old task for backward evaluvation.
                       This is not applicable for forward task since, when you begin a new task you have all the data for new task."""
                    
                    """Backward Accuracy """
                    for t in range( backward_task_samples ):
                        
                        test_x_backward  = prev_tasks_x[t][start_idx: start_idx+test_batch_size] 
                        
                        test_y_backward = prev_tasks_y[t][start_idx: start_idx+test_batch_size]
                       
                        backward_accuracy =  learner.test( test_x_backward, test_y_backward ) 
                        
                        backward_accuracies[test_iter] = backward_accuracies[test_iter] + backward_accuracy
                        
                        overall_accuracies[test_iter] = overall_accuracies[test_iter] + backward_accuracy
                        

                           
                    backward_accuracies[test_iter] = backward_accuracies[test_iter] / backward_task_samples
                    
                    
                    """ Forward Accuracy """
                    test_x_forward, test_y_forward = x[start_idx: start_idx+test_batch_size], y[start_idx: start_idx+test_batch_size]
                    
                    forward_accuracy = learner.test( test_x_forward, test_y_forward) 
                    
                    forward_accuracies[test_iter] = forward_accuracy
                    
                    overall_accuracies[test_iter] =  overall_accuracies[test_iter] + forward_accuracy
                    
                    overall_accuracies[test_iter] =  overall_accuracies[test_iter] / ( backward_task_samples + 1 )
                    
                
                if task_idx % 100 ==0:
                    
                    forward_effective_ranks[test_iter] =  learner.calculate_hessian( test_x_forward, test_y_forward)
                  
                    backward_effective_ranks[test_iter] = learner.calculate_hessian(test_x_backward, test_y_backward)
                else:
                    forward_effective_ranks[test_iter] = forward_effective_ranks[test_iter -1]
                    
                    backward_effective_ranks[test_iter] = backward_effective_ranks[test_iter - 1]
                    
                    
                test_iter += 1
                
                break
            
        prev_tasks_x.append(x.clone())
        
        prev_tasks_y.append(y.clone())
        
        print('Train accuracy: ', train_accuracies[new_iter_start:iter - 1].mean().item(), 
             'Backward accuracy: ', backward_accuracies[ test_iter -1 ].item(),
             'Forward accuracy: ',  forward_accuracies[ test_iter -1 ].item(),
             'Overall accuracy: ', overall_accuracies[ test_iter -1 ].item() ,
            'forward_effective_ranks: ', forward_effective_ranks[ test_iter -1 ].item(),
            'backward_effective_ranks: ', backward_effective_ranks[ test_iter -1 ].item(),
             
             )
        '''
        if task_idx % save_after_every_n_tasks == 0:
            data = {
                  'train_accuracies': train_accuracies.cpu(),
                  'backward_accuracies': backward_accuracies.cpu(),
                  'forward_accuracies': forward_accuracies.cpu(),
                  'overall_accuracies': overall_accuracies.cpu(),
                  'forward_effective_ranks': forward_effective_ranks.cpu(),
                  'backward_effective_ranks': backward_effective_ranks.cpu(),
                  
                   }
            
            result_path = os.path.join( project_root, model_params['model_dir'], 'output.pkl' )
            
            with open(result_path, 'wb+') as f:
                 pickle.dump(data, f)
                 
            model_path = os.path.join(project_root, model_params['model_dir'], 'model.pth' )
           
            torch.save(learner.net.state_dict(), model_path )  
        '''    
       
def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

   
def import_modules():
    
    global Backprop, ContinualBackprop, MyLinear, softmax, DeepFFNN, nll_accuracy, compute_matrix_rank_summaries

    from bp import Backprop
    from cbp import ContinualBackprop
    from linear import MyLinear
    from torch.nn.functional import softmax
    from deep_ffnn import DeepFFNN
    from miscellaneous import nll_accuracy, compute_matrix_rank_summaries


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
    #sys.exit(main(sys.argv[1:]))
    
        
    project_root = os.path.abspath( os.path.join(os.getcwd(), "..","..", ".."))
    
    model_config_path = os.path.join(project_root, "runtime_config","models", "permuted_mnist", "cbp","202.json") 
    
    data_config_path = os.path.join(project_root, "runtime_config","data", "permuted_mnist", "0.json")
    
    sys.exit( main ( ['-c1', model_config_path, '-c2', data_config_path ] ) )
    


    
    