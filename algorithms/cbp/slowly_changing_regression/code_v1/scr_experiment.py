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



import json
import torch

import argparse
import numpy as np
import random
import torch.nn.functional as F


def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

   
def import_modules():    
        
    from algorithms.cbp.slowly_changing_regression.code_v1.data_manager import DataManager 
    from algorithms.cbp.slowly_changing_regression.code_v1.runner import Runner 
    from algorithms.cbp.slowly_changing_regression.code_v1.checkpoint_manager import CheckpointManager 

    #from algorithms.cbp.slowly_changing_regression.code_v1.neural_net import feed_forward_nn
    from algorithms.cbp.slowly_changing_regression.code_v1.AdamGnT import AdamGnT
    from algorithms.cbp.slowly_changing_regression.code_v1.gnt import GnT
    from algorithms.cbp.slowly_changing_regression.code_v1.cbp import ContinualBackprop
    from algorithms.cbp.slowly_changing_regression.code_v1.ffnn import FFNN

    global DataManager, Runner, CheckpointManager, GnT, ContinualBackprop,FFNN
    
    
class TrainContext:
    def __init__(self, num_inputs, num_features, num_outputs, hidden_activation, step_size, weight_decay, beta_1, beta_2):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        net = FFNN(
            input_size=num_inputs,
            num_features=num_features,
            hidden_activation=hidden_activation,
        )
        net.to(self.device)
        
        self.learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt='adam',
            replacement_rate= 0.001,
            decay_rate=0.99,
            device= self.device,
            maturity_threshold= 10,
            util_type='adaptable_contribution',
            init='kaiming',
            accumulate=True,
        )
        
        self.loss = F.mse_loss
        
    
class SCRExperiment:
    
    def __init__(self, config_params):
        
        data_params = config_params["data_config"]
        
        self.data_dir = data_params["data_dir"]
        
        self.num_data_points = data_params["num_data_points"]
        
        self.flip_after =  data_params["flip_after"]
        
        self.num_old_task_window = data_params["num_old_task_window"]
        
        self.num_datapoints_per_timestep = data_params["num_datapoints_per_timestep"]
        
        self.train_size = data_params["train_size"]
        
        
        model_params = config_params["model_config"]
        
        self.model_dir = model_params["model_dir"]
        
        self.num_inputs = model_params["num_inputs"]
        
        self.num_features = model_params['num_features'] 
        
        self.num_outputs = model_params["num_outputs"]
        
        self.hidden_activation = model_params['hidden_activation'] 
        
        self.step_size = model_params["step_size"]
        
        self.weight_decay = model_params["weight_decay"]        
        
        self.beta_1 = model_params["beta_1"]
        
        self.beta_2 = model_params["beta_2"]
        
    def initialize_model(self):
       self.train_context = TrainContext(self.num_inputs, self.num_features, self.num_outputs, self.hidden_activation, self.step_size, self.weight_decay,
                                         self.beta_1, self.beta_2)
    
    def initialize_data_manager(self):
         self.data_manager_obj = DataManager(self.train_context.device, ROOT, self.data_dir, self.flip_after, self.num_data_points,
                                             self.num_old_task_window, self.num_datapoints_per_timestep, self.train_size)
         
    def initialize_runner(self):
        self.runner_obj = Runner(self.num_datapoints_per_timestep)
    
    

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
       
   exp_obj = SCRExperiment(config_params)

   exp_obj.initialize_model()  
    
   exp_obj.initialize_data_manager() 
       
   exp_obj.initialize_runner()

   exp_obj.initialize_checkpoint_manager()
   
   exp_obj.data_manager_obj.create_scr_data()
   
   #exp_obj.checkpoint_obj.load_experiment_checkpoint(exp_obj.train_context, exp_obj.data_manager_obj)
   
   exp_obj.runner_obj.run(exp_obj.train_context, exp_obj.data_manager_obj, exp_obj.checkpoint_obj)
   



if __name__ == '__main__':
    
    config_path = os.path.join( experiment_dir, "configuration.json") 

    sys.exit( main ( ['-c1', config_path ] ) )
    
    
       
