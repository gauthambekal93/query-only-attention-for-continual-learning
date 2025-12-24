# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:06:35 2025

@author: gauthambekal93
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent   # go up two levels, adjust as needed
sys.path.insert(0, str(ROOT))


import json
import torch

import argparse
import numpy as np
import random




def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

   
def import_modules():
    
    from algorithms.full_reinitialization.Code.class_incremental_cifar_100.data_manager import DataManager 
    from algorithms.full_reinitialization.Code.class_incremental_cifar_100.runner import Runner 
    from algorithms.full_reinitialization.Code.class_incremental_cifar_100.checkpoint_manager import CheckpointManager 

    from common.codes.torchvision_modified_resnet import build_resnet18, kaiming_init_resnet_module

    global build_resnet18, kaiming_init_resnet_module, DataManager, Runner, CheckpointManager
    
    
    
class TrainContext:
    def __init__(self, step_size, momentum, weight_decay, total_classes):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.net = build_resnet18(num_classes=total_classes, norm_layer=torch.nn.BatchNorm2d)
        
        self.net.apply(kaiming_init_resnet_module)
        
        self.net.to(self.device)
        
        self.optim = torch.optim.SGD(self.net.parameters(), lr = step_size, momentum= momentum, weight_decay= weight_decay)

        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        
        self.step_size = step_size
        
        self.momentum = momentum
         
        self.weight_decay = weight_decay
       
    
class IncrementalCIFARExperiment:
    
    def __init__(self, data_params, model_params):
        
        """The below line is not a good practice and can lead to silent bugs """
        #if self.device.type == "cuda":    
        #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        
        self.data_dir = data_params["data_dir"]
        
        self.total_classes = data_params["total_classes"]

        self.class_increase_per_task =  data_params["class_increase_per_task"]
        
        self.num_images_per_class = data_params["num_images_per_class"] 
        
        self.initial_num_classes = data_params["initial_num_classes"]
        
        self.running_avg_window = data_params["running_avg_window"]
        
        self.model_dir = model_params["model_dir"]
        
        self.image_dims =  model_params["image_dims"] 
        
        self.batch_sizes =  model_params['batch_sizes']   
        
        self.reset_head =  model_params['reset_head'] 
        
        self.early_stopping = True if "true" == model_params["early_stopping"] else False
        
        self.replacement_rate = model_params["replacement_rate"]
        
        self.utility_function = model_params["utility_function"]
        
        self.maturity_threshold = model_params["maturity_threshold"]
        
        self.noise_std = model_params["noise_std"]
        
        self.perturb_weights_indicator = True if 'true' == model_params["perturb_weights_indicator"] else False
        
        self.step_size = model_params["step_size"]
        
        self.momentum = model_params["momentum"]
                
        self.weight_decay = model_params['weight_decay']
        
        self.num_epochs = model_params['num_epochs'] 
        
        self.model_dir = model_params["model_dir"]
        
        self.train_batch_size = model_params["batch_sizes"]["train"]
        
        self.test_batch_size = model_params["batch_sizes"]["test"]
    
    def initialize_model(self):
       self.train_context = TrainContext(self.step_size, self.momentum, self.weight_decay, self.total_classes)
       
       
    def initialize_data_manager(self):
         self.data_manager_obj = DataManager(root = ROOT, data_dir = self.data_dir, num_images_per_class = self.num_images_per_class , 
                                             initial_num_classes =self.initial_num_classes, 
                                             class_increase_per_task = self.class_increase_per_task, total_classes = self.total_classes,
                                             device = self.train_context.device)
         
    
    def initialize_runner(self):
        self.runner_obj = Runner(self.data_manager_obj, self.num_epochs, train_batch_size = self.train_batch_size, test_batch_size = self.test_batch_size)
    
    

    def initialize_checkpoint_manager(self):
        self.checkpoint_obj = CheckpointManager(self.data_manager_obj, self.runner_obj, root = ROOT, running_avg_window = self.running_avg_window , 
                                                model_dir = self.model_dir )
    
                
                
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
       
   exp_obj = IncrementalCIFARExperiment(data_params, model_params)

   exp_obj.initialize_model()  
    
   exp_obj.initialize_data_manager() 
       
   exp_obj.initialize_runner()

   exp_obj.initialize_checkpoint_manager()
   
   exp_obj.data_manager_obj.create_cifar_data()
   
   
   while exp_obj.data_manager_obj.current_task_id < exp_obj.data_manager_obj.total_tasks:
       
       exp_obj.data_manager_obj.create_task_data()
       
       exp_obj.runner_obj.run(exp_obj.train_context, exp_obj.data_manager_obj, exp_obj.checkpoint_obj)
       
       exp_obj.data_manager_obj.current_task_id += 1
       
       exp_obj.data_manager_obj.current_num_classes += exp_obj.data_manager_obj.class_increase_per_task
       
       exp_obj.initialize_model()


if __name__ == '__main__':
    
    model_config_path = os.path.join( ROOT, "configuration_files","cifar_100", "models", "full_reinitialization", "0.json") 
    
    data_config_path = os.path.join( ROOT, "configuration_files","cifar_100", "data", "0.json")
    
    sys.exit( main ( ['-c1', model_config_path, '-c2', data_config_path ] ) )
    
    
       
