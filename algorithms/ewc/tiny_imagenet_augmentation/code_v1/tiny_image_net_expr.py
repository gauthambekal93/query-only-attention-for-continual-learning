# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:06:35 2025

@author: gauthambekal93
"""

import os
import sys
from pathlib import Path
experiment_dir = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent   # go up two levels, adjust as needed
sys.path.insert(0, str(ROOT))

# Get current file's directory
#BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

# Add it to sys.path
#sys.path.append(str(BASE_DIR / "common" / "codes"))
#sys.path.append(str(BASE_DIR / "algorithms" / "ewc"/ "Code"/"split_image_net"))


import json
import torch

import argparse
import numpy as np
import random
from tqdm import tqdm



def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

   
def import_modules():    
        
    from algorithms.ewc.tiny_imagenet_augmentation.code_v1.data_manager import DataManager 
    from algorithms.ewc.tiny_imagenet_augmentation.code_v1.runner import Runner 
    from algorithms.ewc.tiny_imagenet_augmentation.code_v1.checkpoint_manager import CheckpointManager 

    from algorithms.ewc.tiny_imagenet_augmentation.code_v1.torchvision_modified_resnet import build_resnet18, kaiming_init_resnet_module

    global build_resnet18, kaiming_init_resnet_module, DataManager, Runner, CheckpointManager
    
    
class TrainContext:
    def __init__(self, step_size, momentum, weight_decay, classes_per_task):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.net = build_resnet18(num_classes=classes_per_task, norm_layer=torch.nn.BatchNorm2d)
        
        self.net.apply(kaiming_init_resnet_module)
        
        self.net.to(self.device)
        
        self.opt = torch.optim.SGD(self.net.parameters(), lr = step_size, momentum= momentum, weight_decay= weight_decay)

        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        
        self.net.initialize_fisher()
    
class Incremental_Tiny_Imagenet_Experiment:
    
    def __init__(self, config_params):
        
        
        data_params = config_params["data_config"]
        
        self.data_dir = data_params["data_dir"]
        
        self.num_tasks = data_params["num_tasks"]
    
        self.total_classes = data_params["total_classes"]
        
        self.classes_per_task = data_params["classes_per_task"]
        
        self.num_old_task_window = data_params["num_old_task_window"]
        
        self.num_datapoints_per_timestep =  data_params['num_datapoints_per_timestep']   


        
        model_params = config_params["model_config"]
        
        self.model_dir = model_params["model_dir"]
        
        self.step_size = model_params["step_size"]
                
        self.weight_decay = model_params['weight_decay']
        
        self.momentum = model_params["momentum"]
        
        self.ewc_lambda = model_params["ewc_lambda"]
        
        
        
    def initialize_model(self):
       self.train_context = TrainContext(self.step_size, self.momentum, self.weight_decay, self.classes_per_task)
       
    
    def initialize_data_manager(self):
         self.data_manager_obj = DataManager(self.train_context.device, ROOT, self.data_dir, self.classes_per_task, 
                                             self.total_classes, self.num_old_task_window, self.num_datapoints_per_timestep,
                                             self.num_tasks)
         

    def initialize_runner(self):
        self.runner_obj = Runner(self.num_datapoints_per_timestep, self.ewc_lambda)
    
    

    def initialize_checkpoint_manager(self):
        self.checkpoint_obj = CheckpointManager(self.data_manager_obj, self.runner_obj, root = ROOT, model_dir = self.model_dir  )
    

    
def main(arguments):
   parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
   parser.add_argument('-c1', help="Path to the file containing the parameters for the experiment", type=str)
   parser.add_argument('-c2', help="Path to the file containing the parameters for the experiment", type=str)
   
   args = parser.parse_args(arguments)
  
   with open(args.c1, 'r') as f:
       config_params = json.load(f)
      
   set_seed(config_params["model_config"]["seed"])
    
   import_modules()
       
   exp_obj = Incremental_Tiny_Imagenet_Experiment(config_params)

   exp_obj.initialize_model()  
    
   exp_obj.initialize_data_manager() 
       
   exp_obj.initialize_runner()

   exp_obj.initialize_checkpoint_manager()
   
   #exp_obj.data_manager_obj.create_tiny_imagenet_data()

   exp_obj.data_manager_obj.load_tiny_imagenet_data()
   
   exp_obj.runner_obj.run(exp_obj.train_context, exp_obj.data_manager_obj, exp_obj.checkpoint_obj)
   



if __name__ == '__main__':
    
    config_path = os.path.join( experiment_dir, "configuration.json") 

    sys.exit( main ( ['-c1', config_path ] ) )
    
    
       
