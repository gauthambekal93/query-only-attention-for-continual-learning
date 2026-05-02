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
#sys.path.append(str(BASE_DIR / "analysis_and_workings" / "bp"/ "Code"/"split_image_net"))


import json
import torch

import argparse
import numpy as np
import random
#from tqdm import tqdm



def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

   
def import_modules():
    
    from analysis_and_workings.full_attention.permuted_mnist.parameter_adaptations.data_manager import DataManager 
    from analysis_and_workings.full_attention.permuted_mnist.parameter_adaptations.runner import Runner 
    from analysis_and_workings.full_attention.permuted_mnist.parameter_adaptations.checkpoint_manager import CheckpointManager 
    from analysis_and_workings.full_attention.permuted_mnist.parameter_adaptations.neural_networks import ERNetwork
    
    global  ERNetwork, DataManager, Runner, CheckpointManager
    
    
    
class TrainContext:
    def __init__(self, input_size, num_features, classes_per_task, num_attention_layers, step_size, weight_decay, total_classes):

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.net = ERNetwork(input_size=input_size, num_features=num_features, num_outputs=classes_per_task)
        
        self.net.to(self.device)
        
        beta_1, beta_2 = 0.9,  0.999
        
        self.opt = torch.optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
        
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        
        
    
class PermutedMNISTExperiment:
    
    def __init__(self,config_params):
        
        data_params = config_params["data_config"]
        
        self.data_dir = data_params["data_dir"]
        
        self.num_tasks = data_params["num_tasks"]
    
        self.total_classes = data_params["total_classes"]
        
        self.classes_per_task = data_params["classes_per_task"]
        
        self.num_old_task_window = data_params["num_old_task_window"]
        
        self.num_datapoints_per_timestep =  data_params['num_datapoints_per_timestep']   
        
        self.change_after = data_params["change_after"]
        
        
        model_params = config_params["model_config"]
        
        self.model_dir = model_params["model_dir"]
        
        self.input_size =  model_params["input_size"] 
        
        self.num_features = model_params["num_features"]
        
        self.num_attention_layers = model_params["num_attention_layers"]
        
        self.step_size = model_params["step_size"]
                
        self.weight_decay = model_params['weight_decay']
    
        self.buffer_size = model_params["buffer_size"]
                
        self.test_batch_size = model_params["test_batch_size"]
        
        self.samples_per_label = model_params["samples_per_label"]

        

        
    def initialize_model(self):
         self.train_context =  TrainContext(self.input_size, self.num_features, self.classes_per_task, self.num_attention_layers, self.step_size, self.weight_decay, self.total_classes)
        
      
    def initialize_data_manager(self):
         self.data_manager_obj = DataManager(self.train_context.device, ROOT, self.data_dir, self.classes_per_task, 
                                             self.num_old_task_window, self.buffer_size, self.num_datapoints_per_timestep, self.samples_per_label, self.num_tasks)
         
    
    def initialize_runner(self):
        self.runner_obj = Runner(self.num_datapoints_per_timestep , self.test_batch_size)
    
    

    def initialize_checkpoint_manager(self):
        self.checkpoint_obj = CheckpointManager(self.data_manager_obj, self.runner_obj, root = ROOT ,  model_dir = self.model_dir )
    


def main(arguments):
   parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
   parser.add_argument('-c1', help="Path to the file containing the parameters for the experiment", type=str)
   parser.add_argument('-c2', help="Path to the file containing the parameters for the experiment", type=str)
   
   args = parser.parse_args(arguments)
  
   with open(args.c1, 'r') as f:
      config_params = json.load(f)
      
   set_seed(config_params["model_config"]["seed"])
    
   import_modules()
       
   exp_obj = PermutedMNISTExperiment(config_params)

   exp_obj.initialize_model()  
    
   exp_obj.initialize_data_manager() 
       
   exp_obj.initialize_runner()

   exp_obj.initialize_checkpoint_manager()
   
   exp_obj.data_manager_obj.create_permute_mnist_data()
   
   #exp_obj.checkpoint_obj.load_experiment_checkpoint(exp_obj.train_context, exp_obj.data_manager_obj)
   
   exp_obj.runner_obj.run(exp_obj.train_context, exp_obj.data_manager_obj, exp_obj.checkpoint_obj)
   



if __name__ == '__main__':
    
    config_path = os.path.join( experiment_dir, "configuration-2.json") 

    sys.exit( main ( ['-c1', config_path ] ) )
  
    
    
       
