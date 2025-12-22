# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 20:54:38 2025

@author: gauthambekal93
"""



import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent   # go up two levels, adjust as needed
sys.path.insert(0, str(ROOT))

# Get current file's directory
#BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

# Add it to sys.path
#sys.path.append(str(BASE_DIR / "common" / "codes"))
#sys.path.append(str(BASE_DIR / "algorithms" / "bp"/ "Code"/"split_image_net"))


import json
import torch

import argparse
import numpy as np
import random
from tqdm import tqdm

#from collections import deque
#import time


def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

   
def import_modules():
    
    from algorithms.bp.Code.class_incremental_cifar_100.experiment_data import CifarData #get_data_model, get_task_data, create_result_dir

    from common.codes.torchvision_modified_resnet import build_resnet18, kaiming_init_resnet_module

    global build_resnet18, kaiming_init_resnet_module, CifarData
    

class IncrementalCIFARExperiment():
    
    def __init__(self, data_params, model_params):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        """The below line is not a good practice and can lead to silent bugs """
        #if self.device.type == "cuda":    
        #    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        self.data_model = CifarData( ROOT, data_params, model_params)    
        
        self.total_classes = data_params["total_classes"]

        self.class_increase_per_task =  data_params["class_increase_per_task"]
        
        self.total_tasks = self.total_classes / self.class_increase_per_task
        
        self.current_num_classes = self.class_increase_per_task
        
        self.num_images_per_class = data_params["num_images_per_class"] #450
        
        self.class_increase_frequency = data_params["class_increase_frequency"] #450
        
        self.image_dims =  model_params["image_dims"] #(32, 32, 3)
        
        self.batch_sizes =  model_params['batch_sizes']   # {"train": 90, "test": 100, "validation":50}
        
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
        
        self.epochs_per_task = int( model_params['num_epochs']  / self.total_tasks )
        
        self.model_dir = model_params["model_dir"]
        
        self.net = build_resnet18(num_classes=self.total_classes, norm_layer=torch.nn.BatchNorm2d)
        
        self.net.apply(kaiming_init_resnet_module)

        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.step_size, momentum=self.momentum, weight_decay=self.weight_decay)

        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
       
        self.net.to(self.device)
        
        self.current_task_id = 0
        
        
        
    def evaluvate_network(self, epoch):
        
        avg_loss = 0.0
        avg_acc = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            test_ids = torch.range( len(self.data_model.task_test_y) )
            
            for batch_no in range( len(test_ids) ):
                
                batch_ids = test_ids[batch_no: batch_no + self.batch_sizes['test']]
            
                batch_x, batch_y = self.data_model.task_test_x[batch_ids].to(self.device), self.data_model.task_test_y[batch_ids].to(self.device)
                
                predictions = self.net.forward(batch_x )[:, self.all_classes[:self.current_num_classes]]
                
                avg_loss += self.loss(predictions, batch_y)
                
                avg_acc += torch.mean((predictions.argmax(axis=1) == batch_y.argmax(axis=1)).to(torch.float32))
                
                num_test_batches += 1
            
        self.data_model.save_test(avg_loss / num_test_batches, avg_acc / num_test_batches, epoch )
         
        
 
    def train(self):

        """train model """
        for epoch in tqdm( range(self.epochs_per_task)):
            
            self.net.train()
            
            rand_idx = torch.randperm(len(self.data_model.task_train_y)) 
            
            for batch_no in range( 0, len(rand_idx), self.batch_sizes['train'] ):
                
                batch_ids = rand_idx[batch_no: batch_no + self.batch_sizes['train']]
            
                batch_x, batch_y = self.data_model.task_train_x[batch_ids].to(self.device), self.data_model.task_train_y[batch_ids].to(self.device)
                
                # reset gradients
                for param in self.net.parameters(): 
                    param.grad = None   # apparently faster than optim.zero_grad()
                
                predictions = self.net.forward(batch_x )   [ self.data_model.label_ids_flattened [:self.current_num_classes] ] 
                
                current_reg_loss = self.loss(predictions, batch_y)
                
                current_reg_loss.backward()
                
                self.optim.step()
                
                self.data_model.running_accuracy += torch.mean((predictions.argmax(axis=1) == batch_y.argmax(axis=1)).to(torch.float32)).detach()
                self.data_model.running_loss += current_reg_loss.detach()
                
                """save checkpoints """
                if (batch_no + 1) % self.data_model.running_avg_window == 0:
                    self.data_model.save_train(self.net)
            
            self.net.eval()
            """obtain performance """
            self.evaluvate_network()
            
            
            
    def run(self):
        
        """iterate over tasks """
        while self.current_task_id < self.total_tasks:

            self.data_model.create_cifar_data()
        
            self.data_model.create_result_dir()
            
            self.data_model.create_task_data(self.current_task_id)
            
            self.train()
            
            self.current_task_id += 1
            
            self.current_num_classes += self.class_increase_per_task
    


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
    
   model = IncrementalCIFARExperiment(data_params, model_params)
   
   model.run()




if __name__ == '__main__':
    
    
    model_config_path = os.path.join( ROOT, "configuration_files","cifar_100", "models", "bp", "0.json") 
    
    data_config_path = os.path.join( ROOT, "configuration_files","cifar_100", "data", "0.json")
    
    sys.exit( main ( ['-c1', model_config_path, '-c2', data_config_path ] ) )
 