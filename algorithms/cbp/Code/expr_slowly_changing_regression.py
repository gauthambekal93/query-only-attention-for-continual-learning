import os
import sys

from pathlib import Path
# Get current file's directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

# Add it to sys.path
sys.path.append(str(BASE_DIR / "common" / "codes"))


import json
import pickle
import argparse
import copy



import numpy as np
import random
import torch



def expr(model_params , data_params):
    agent_type = model_params['agent']
    data_file_path = os.path.join(project_root ,data_params['data_dir'] )   #this is the path where data for training the model is located
    num_data_points = int(model_params['num_data_points']) # actual datapoints we use to train the model, can be different from 1000000
    to_log = False
    to_log_grad = False
    to_log_activation = False
    beta_1 = 0.9
    beta_2 = 0.999
    weight_decay = 0.0
    accumulate = False
    perturb_scale = 0
    if 'to_log' in model_params.keys():
        to_log = model_params['to_log']
    if 'to_log_grad' in model_params.keys():
        to_log_grad = model_params['to_log_grad']
    if 'to_log_activation' in model_params.keys():
        to_log_activation = model_params['to_log_activation']
    if 'beta_1' in model_params.keys():
        beta_1 = model_params['beta_1']
    if 'beta_2' in model_params.keys():
        beta_2 = model_params['beta_2']
    if 'weight_decay' in model_params.keys():
        weight_decay = model_params['weight_decay']
    if 'accumulate' in model_params.keys():
        accumulate = model_params['accumulate']
    if 'perturb_scale' in model_params.keys():
        perturb_scale = model_params['perturb_scale']

    num_inputs = model_params['num_inputs'] #20 
    num_features = model_params['num_features'] # 5, This is hidden layer size
    hidden_activation = model_params['hidden_activation'] #relu
    step_size = model_params['step_size'] #0.01 is the learning rate
    opt = model_params['opt']             #sgd optimizer
    replacement_rate = model_params["replacement_rate"] # this value is used only in cbp and not in bp
    decay_rate = model_params["decay_rate"] #0
    mt = 10                           #maturity threshold # this value is used only in cbp and not in bp
    util_type='adaptable_contribution'
    init = 'kaiming'
    if "mt" in model_params.keys():
        mt = model_params["mt"]
    if "util_type" in model_params.keys():
        util_type = model_params["util_type"]
    if "init" in model_params.keys():
        init = model_params["init"]

    if agent_type == 'linear':
        net = MyLinear(
            input_size=num_inputs,
        )
    else:
        net = FFNN(
            input_size=num_inputs,
            num_features=num_features,
            hidden_activation=hidden_activation,
        )

    if agent_type == 'bp' or agent_type == 'linear' or agent_type == 'l2':
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            beta_1=beta_1,  #beta_1 and beta_2 not used in sgd
            beta_2=beta_2,
            weight_decay=weight_decay,
            to_perturb=(perturb_scale > 0),  #this is by default false
            perturb_scale=perturb_scale,
        )
    elif agent_type == 'cbp':  #cbp is continuouse back prop
        learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt=opt,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            device='cpu',
            maturity_threshold=mt,
            util_type=util_type,
            init=init,
            accumulate=accumulate,
        )

    with open(data_file_path, 'rb+') as f:  #get the input and output features for training  inputs.shape torch.Size([10010000, 20]), outputs.shape torch.Size([10010000, 1])
        inputs, outputs, _ = pickle.load(f)  

    errs, forward_test_errs = [],  []
    if to_log: weight_mag = torch.zeros((num_data_points, 2), dtype=torch.float)
    if to_log_grad: grad_mag = torch.zeros((num_data_points, 2), dtype=torch.float)
    if to_log_activation: activation = torch.zeros((num_data_points, ), dtype=torch.float)
    
    
    datapoints_per_task = data_params['flip_after'] 
    
    total_tasks = int(  num_data_points/ datapoints_per_task )
    
    train_errors = []
    
    backward_errors =  torch.zeros( total_tasks , dtype=torch.float)  
    
    forward_errors =  torch.zeros( total_tasks , dtype=torch.float)
    
    overall_errors =  torch.zeros( total_tasks , dtype=torch.float)
    
    forward_effective_ranks = torch.zeros( total_tasks , dtype=torch.float)
    
    backward_effective_ranks = torch.zeros( total_tasks , dtype=torch.float)
    
    train_size, test_size, task_num, past_task_offset = 9000, 1000, 0 , 5
    
      
    i = 0
    
    while i < num_data_points: 
        
        train_end_index = task_num* datapoints_per_task + train_size
        
        if ( i < train_end_index )  or ( task_num < past_task_offset ):
            mode = "Train"
        else:
            mode = "Test"
            
        if  mode == "Train":
                       
            x, y = inputs[i: i+1], outputs[i: i+1]
            
            train_errors.append( learner.learn(x=x, target=y ) ) 
            
        else :
            
            with torch.no_grad():
                
                x, y = inputs[ i : i + test_size], outputs[ i : i + test_size]
                
                x_copy, y_copy  = copy.deepcopy(x), copy.deepcopy(y)
                
                forward_error = learner.test(x = x, target = y )
                
                for j in range(past_task_offset):
                    
                    start, end  = i -1 * ( j +1) * datapoints_per_task ,  i -1* ( j + 1) * datapoints_per_task + test_size
                    
                    x, y = inputs[ start : end ], outputs[ start: end ]  
                    
                    backward_error = learner.test(x=x, target=y )
                    
                    backward_errors[ task_num ] = backward_errors[ task_num ] + backward_error
                    
                    overall_errors[ task_num ] = overall_errors[task_num] + backward_error
                
                    if j ==past_task_offset-1:
                        x_copy_2, y_copy_2  = copy.deepcopy(x), copy.deepcopy(y)
                        
                backward_errors[task_num] = backward_errors[task_num] / past_task_offset
                
                forward_errors[task_num] = forward_error
                
                overall_errors[task_num] =  overall_errors[task_num] + forward_error
                
                overall_errors[task_num] =  overall_errors[task_num] / ( past_task_offset + 1 )
        
            forward_effective_ranks[task_num] = learner.calculate_hessian(x_copy, y_copy)
            
            backward_effective_ranks[task_num] = learner.calculate_hessian(x_copy_2, y_copy_2)
            
        #if ( task_num % 10 == 0 ) and ( task_num > 0 ):
        if ( i % datapoints_per_task ==0 ) and (task_num > 0):          
             data = {
                      'train mse': torch.tensor(train_errors).cpu(),
                      'backward task mse': backward_errors[ past_task_offset : ].cpu(),
                      'forward task mse': forward_errors[ past_task_offset : ].cpu(),
                      'overall task mse': overall_errors[ past_task_offset : ].cpu(),
                      'forward effective rank': forward_effective_ranks[ past_task_offset : ].cpu(),
                      'backward effective rank': backward_effective_ranks[ past_task_offset : ].cpu()
                       }
             
             result_path = os.path.join( project_root, model_params['model_dir'], 'output.pkl' )
             
             with open(result_path, 'wb+') as f:
                 pickle.dump(data, f)
                 
             
             model_path = os.path.join(project_root, model_params['model_dir'], 'model.pth' )
               
             torch.save(learner.net.state_dict(), model_path )  
        
        

        if mode =="Train":
            i = i + 1 
        else:
            i = i + test_size
        
        
        if ( i % datapoints_per_task ==0 ) and ( i >= datapoints_per_task):
            
            print("Index: ", i, "Task Num: ", task_num)
            
            print('train_mse: ', torch.tensor( train_errors[-train_size : ] ).mean().item(), 
                  'backward_task_mse: ', backward_errors[ task_num ].item(),
                  'forward_task_mse: ',  forward_errors[ task_num ].item(),
                  'overall_task_mse: ', overall_errors[ task_num ].item(),
                  'forward_effective_ranks: ', forward_effective_ranks[ task_num ].item(),
                  'backward_effective_ranks: ', backward_effective_ranks[ task_num ].item(),)
            
            task_num += 1    
            
                
    data = {
             'train mse': torch.tensor(train_errors).cpu(),
             'backward task mse': backward_errors[ past_task_offset : ].cpu(),
             'forward task mse': forward_errors[ past_task_offset : ].cpu(),
             'overall task mse': overall_errors[ past_task_offset : ].cpu(),
             'forward effective rank': forward_effective_ranks[ past_task_offset : ].cpu(),
             'backward effective rank': backward_effective_ranks[ past_task_offset : ].cpu()
              }
    
    result_path = os.path.join( project_root, model_params['model_dir'], 'output.pkl' )
    
    with open(result_path, 'wb+') as f:
        pickle.dump(data, f)
        
    
    model_path = os.path.join(project_root, model_params['model_dir'], 'model.pth' )
      
    torch.save(learner.net.state_dict(), model_path )  
       
def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

   
def import_modules():
    global FFNN, MyLinear, Backprop, ContinualBackprop
    
    from ffnn import FFNN
    from linear import MyLinear
    from bp import Backprop
    from cbp import ContinualBackprop
    
    
    
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
    
    model_config_path = os.path.join(project_root, "runtime_config","models", "slowly_changing_regression", "cbp","202.json")
    
    data_config_path = os.path.join(project_root, "runtime_config","data", "slowly_changing_regression", "0.json")
    
    sys.exit( main ( ['-c1', model_config_path, '-c2', data_config_path ] ) )
    
