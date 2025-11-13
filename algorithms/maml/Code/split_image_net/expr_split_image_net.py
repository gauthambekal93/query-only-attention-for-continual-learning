# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 20:54:38 2025

@author: gauthambekal93
"""



import os
import sys
sys.path.append("C:/Users/gauthambekal93/Research/continual_learning/loss_of_plasticity_and_forgetting/common/codes")
sys.path.append("C:/Users/gauthambekal93/Research/continual_learning/loss_of_plasticity_and_forgetting/algorithms/maml/Code/split_image_net")


import json
import torch
import pickle
import argparse
import numpy as np
import random
from tqdm import tqdm

from collections import deque




def expr(model_params , data_params):
    
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if dev == torch.device("cuda"):    
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    
    """-----THESE ARE DATA SPECIFIC PARAMETERS----- """

    num_tasks = data_params['num_tasks']
    
    task_datapoints = data_params["task_datapoints"]
    
    images_per_class_train = data_params["images_per_class_train"]

    images_per_class_val = data_params["images_per_class_val"]

    total_labels = data_params["total_labels"]

    classes_per_task =  data_params["classes_per_task"]
        
    dest_data_dir = data_params['dest_data_dir']
    
    save_after_every_n_tasks = data_params["save_after_every_n_tasks"]
    
    """-----THESE ARE MODEL SPECIFIC PARAMETERS----- """
    
    if "use_gpu" in model_params.keys():
        use_gpu = model_params["use_gpu"]
        
    if "inner_step_size" in model_params.keys():
        inner_step_size = model_params["inner_step_size"]
    
    if "outer_step_size" in model_params.keys():
        outer_step_size = model_params["outer_step_size"]
        
    if "momentum" in model_params.keys():
        momentum = model_params["momentum"]
            
    if 'weight_decay' in model_params.keys():
         weight_decay = model_params['weight_decay']
         
    if "beta_1" in model_params.keys():
        beta_1 = model_params["beta_1"]
                
    if "beta_2" in model_params.keys():
        beta_2 = model_params["beta_2"]
    
    if "act_type" in model_params.keys():
        act_type = model_params["act_type"]
             
    if "loss" in model_params.keys():
        loss = model_params["loss"]
    
    if "opt" in model_params.keys():
        opt = model_params["opt"]
                
    if 'num_features' in model_params.keys():
        num_features = model_params['num_features']
    
    if "num_outputs" in model_params.keys():
        num_outputs = model_params["num_outputs"]    
            
    if 'num_hidden_layers' in model_params.keys():
        num_hidden_layers = model_params['num_hidden_layers']

    if 'change_after' in model_params.keys():
        change_after = model_params['change_after']
        
    if 'mini_batch_size' in model_params.keys():
        mini_batch_size = model_params['mini_batch_size']

    if "queries_per_mini_batch" in model_params.keys():
        queries_per_mini_batch = model_params["queries_per_mini_batch"]

    if "past_task_offset" in model_params.keys():
        past_task_offset = model_params["past_task_offset"]
        
    if "model_dir" in model_params.keys():
        model_dir = model_params["model_dir"]
    
    if "buffer_depth" in model_params.keys():
        buffer_depth = model_params["buffer_depth"]
        
    if "num_support" in model_params.keys():
        num_support = model_params["num_support"]
    
    if "num_query" in model_params.keys():
        num_query = model_params["num_query"]
        
    if "num_of_task_sampled" in model_params.keys():
        num_of_task_sampled = model_params["num_of_task_sampled"]
            
    if "inner_loop_count" in model_params.keys():
        inner_loop_count = model_params["inner_loop_count"]
        
    total_examples = int(num_tasks * change_after)
    
    total_iters = int(total_examples/mini_batch_size)

    iter , test_iter = 0, 0
    
    with open(os.path.join(project_root, dest_data_dir  ), 'rb+') as f:
        image_net_train_x, image_net_train_y, _, image_net_val_x, image_net_val_y = pickle.load(f)
        if use_gpu == 0:
            image_net_train_x = image_net_train_x.to(dev)
            image_net_train_y = image_net_train_y.to(dev)
            
            image_net_val_x = image_net_val_x.to(dev)
            image_net_val_y = image_net_val_y.to(dev)        
            
    label_ids = [i for i in range(0, total_labels)]      
    
    prev_task_data = deque(maxlen = past_task_offset)
    
    buffer = Replay_Buffer(buffer_depth, classes_per_task, num_support, num_query, num_of_task_sampled, queries_per_mini_batch, dev )
    
    """We just hardcoded stepsize for testing only """
    # input_features is  128 * 3 * 3 is because we flatten the cnn output
    maml = MAML(input_features= 128 * 3 * 3 , num_features=num_features, num_outputs=num_outputs, num_hidden_layers=num_hidden_layers, 
                          act_type=act_type, opt = opt, inner_step_size= inner_step_size, outer_step_size = outer_step_size , 
                          weight_decay= weight_decay,  momentum= momentum, beta_1=beta_1, beta_2=beta_2, loss=loss, 
                          task_datapoints = task_datapoints, classes_per_task = classes_per_task, 
                          inner_loop_count = inner_loop_count, num_of_task_sampled = num_of_task_sampled).to(dev)
    
    train_accuracies = torch.zeros(total_iters, dtype=torch.float)
    
    backward_accuracies =  torch.zeros( num_tasks, dtype=torch.float)  
    
    forward_accuracies =  torch.zeros( num_tasks , dtype=torch.float)
        
    overall_accuracies =  torch.zeros( num_tasks , dtype=torch.float)
    
    
    for task_idx in (range(num_tasks)):
        
        print("Task Index ", task_idx)
                
        new_iter_start = iter 
        
        train_x, train_y, train_y_one_hot, label_1, label_2 = generate_train_data(image_net_train_x, label_ids, images_per_class_train, classes_per_task)
        
        for start_idx in tqdm(range(0, change_after, mini_batch_size)):
          
            buffer.add_new_data( train_x [start_idx : start_idx + mini_batch_size], 
                                 train_y [start_idx : start_idx + mini_batch_size], 
                                 train_y_one_hot [start_idx: start_idx + mini_batch_size] )
            
            buffer.delete_old_data()
            
            if len(buffer.buffer_x)>= buffer_depth:
                
                support_dict, query_dict = buffer.sample_task_data()
                
                train_loss, train_accuracy = maml.learn(support_dict, query_dict) 
            
                train_accuracies[iter] = train_accuracy
             
                iter += 1   
        
        
        data_dict_val = generate_test_data(train_x, train_y_one_hot, image_net_val_x, image_net_val_y, label_1, label_2 , images_per_class_val, classes_per_task, num_support )
        
        
        if  task_idx >= past_task_offset:
                
            forward_accuracies[test_iter] = maml.test( data_dict_val ) 
            
            backward_accuracies[test_iter] =  maml.test( prev_task_data[0] ) 
                
            overall_accuracies[test_iter]  = (forward_accuracies[test_iter] + backward_accuracies[test_iter] ) / 2
        
        print("Task ID ", task_idx, 
              'Train accuracy: ', train_accuracies[new_iter_start:iter - 1].mean().item(), 
              'Forward Test accuracy: ', forward_accuracies[test_iter].item(),
              'Backward Test accuracy: ', backward_accuracies[test_iter].item(),
              'Overall Test accuracy: ', overall_accuracies[test_iter].item()
              )
        
        prev_task_data.append( data_dict_val )
        
        test_iter += 1
        
        if task_idx % save_after_every_n_tasks == 0:
            data = {
                   'train_accuracies': train_accuracies.cpu(),
                   'backward_accuracies': backward_accuracies.cpu(),
                   'forward_accuracies': forward_accuracies.cpu(),
                   'overall_accuracies': overall_accuracies.cpu(),
                 }
            result_path = os.path.join( project_root, model_dir, 'output.pkl' )
            
            with open(result_path, 'wb+') as f:
                 pickle.dump(data, f)
                 
            model_path = os.path.join(project_root, model_dir, 'model.pth' )
           
            torch.save(maml.state_dict(), model_path ) 
        
        
        
    print("stop")
    print("stop")     
    print("stop")


def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

   
def import_modules():
    
    from replay_buffer_split_image_net import Replay_Buffer
    from relation_network_split_image_net import MAML
    from task_data_generator import generate_train_data, generate_test_data
    
    global Replay_Buffer, MAML, generate_train_data, generate_test_data



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
    
    project_root = os.path.abspath( os.path.join(os.getcwd(), "..","..", "..", ".."))
    
    model_config_path = os.path.join(project_root, "configuration_files", "split_image_net", "models","maml", "0.json") 
    
    data_config_path = os.path.join(project_root, "configuration_files","split_image_net", "data", "0.json")
    
    sys.exit( main ( ['-c1', model_config_path, '-c2', data_config_path ] ) )
 