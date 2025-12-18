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
import pickle
import argparse
import numpy as np
import random
from tqdm import tqdm

from collections import deque
import time


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
        
        if self.device.type == "cuda":    
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        self.data_model = CifarData( ROOT, data_params, model_params)    
        
        self.total_classes = data_params["total_classes"]

        self.classes_per_task =  data_params["classes_per_task"]
        
        self.total_tasks = self.total_classes / self.classes_per_task
        
        self.current_num_classes = self.classes_per_task
        
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
        
        self.num_epochs = model_params['num_epochs']  
        
        self.model_dir = model_params["model_dir"]
        
        self.net = build_resnet18(num_classes=self.total_classes, norm_layer=torch.nn.BatchNorm2d)
        
        self.net.apply(kaiming_init_resnet_module)

        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.step_size, momentum=self.momentum, weight_decay=self.weight_decay)

        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
       
        self.net.to(self.device)
        
        self.current_epoch = 0
        
        self.current_task_id = 0
        
        self.train_summary, self.test_summary = {}, {} 
        

        
    def save_test(self, current_accuracy, current_reg_loss):
        
        self.test_summary["accuracy"] = current_accuracy.detach()
        self.test_summary["loss"] = current_reg_loss.detach()
        
    def evaluvate_network(self):
        
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
            
        
        self.save_test(avg_loss / num_test_batches, avg_acc / num_test_batches)
         
    
    
    def save_train(self, current_accuracy, current_reg_loss):
        
        self.train_summary["accuracy"] = current_accuracy.detach()
        self.train_summary["loss"] = current_reg_loss.detach()
        
    def train(self):

        """train model """
        for epoch in tqdm(range(self.current_epoch, self.num_epochs)):
            
            rand_idx = torch.random.permutation( len(self.data_model.task_train_y) )
            
            for batch_no in range( len(rand_idx) ):
                
                batch_ids = rand_idx[batch_no: batch_no + self.batch_sizes['train']]
            
                batch_x, batch_y = self.data_model.task_train_x[batch_ids].to(self.device), self.data_model.task_train_y[batch_ids].to(self.device)
                
                # reset gradients
                for param in self.net.parameters(): 
                    param.grad = None   # apparently faster than optim.zero_grad()
                
                predictions = self.net.forward(batch_x, current_features =[] )[:, self.all_classes[:self.current_num_classes]]
                
                current_reg_loss = self.loss(predictions, batch_y)
                
                current_reg_loss.backward()
                
                self.optim.step()
                
                current_accuracy = torch.mean((predictions.argmax(axis=1) == batch_y.argmax(axis=1)).to(torch.float32))
        
        self.current_epoch += 1
        
        """save checkpoints """
        self.save_train(current_accuracy, current_reg_loss)
        
        """obtain performance """
        self.evaluvate_network()
        
        
    def run(self):
        
        """iterate over tasks """
        while self.current_task_id < self.total_tasks:

            self.data_model.create_cifar_data()
        
            self.data_model.create_result_dir()
            
            self.data_model.create_task_data()
            
            self.train()
            
            self.current_task_id = self.current_task_id + 1
        
    
def expr(model_params , data_params):
    

    
    
    """-----THESE ARE DATA SPECIFIC PARAMETERS----- """

    #num_tasks = data_params['num_tasks']
    
    #task_datapoints = data_params["task_datapoints"]
    
    #images_per_class_train = data_params["images_per_class_train"]

    #images_per_class_val = data_params["images_per_class_val"]

    
        
    #dest_data_dir = data_params['dest_data_dir']
    
    #save_after_every_n_tasks = data_params["save_after_every_n_tasks"]
    
    """-----THESE ARE MODEL SPECIFIC PARAMETERS----- """
    

        
    


    

            
            
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
    
    # input_features is  128 * 3 * 3 is because we flatten the cnn output.
    mlp = MLP(input_features= 128 * 3 * 3   , num_features=num_features, num_outputs=num_outputs, 
                          act_type=act_type, opt = opt, step_size = step_size , weight_decay= weight_decay,  momentum= momentum, 
                          beta_1=beta_1, beta_2=beta_2, loss=loss, task_datapoints = task_datapoints, classes_per_task = classes_per_task).to(dev)
    
    train_accuracies = torch.zeros(total_iters, dtype=torch.float)
    
    backward_accuracies =  torch.zeros( num_tasks, dtype=torch.float)  
    
    forward_accuracies =  torch.zeros( num_tasks , dtype=torch.float)
        
    overall_accuracies =  torch.zeros( num_tasks , dtype=torch.float)
    
    
    for task_idx in (range(num_tasks)):
        
        print("Task Index ", task_idx)
                
        new_iter_start = iter 
        
        train_x, train_y, train_y_one_hot, label_1, label_2 = generate_train_data(image_net_train_x, label_ids, images_per_class_train, classes_per_task)
        
        for start_idx in tqdm(range(0, change_after, mini_batch_size)):
                
            train_loss, train_accuracy = mlp.learn( train_x [start_idx : start_idx + mini_batch_size] , train_y_one_hot [start_idx: start_idx + mini_batch_size]) 
        
            train_accuracies[iter] = train_accuracy
         
            iter += 1   
          
        data_dict_val = generate_test_data(image_net_val_x, image_net_val_y, label_1, label_2 , images_per_class_val, classes_per_task )
        
        if  task_idx >= past_task_offset:
            
            with torch.no_grad():
                
                forward_accuracies[test_iter] = mlp.test( data_dict_val ) 
                
                backward_accuracies[test_iter] =  mlp.test( prev_task_data[0] ) 
                
            overall_accuracies[test_iter]  = (forward_accuracies[test_iter] + backward_accuracies[test_iter] ) / 2
        
        print("Task ID ", task_idx, 
              'Train accuracy: ', train_accuracies[new_iter_start:iter - 1].mean().item(), 
              'Forward Test accuracy: ', forward_accuracies[test_iter].item(),
              'Backward Test accuracy: ', backward_accuracies[test_iter].item(),
              'Overall Test accuracy: ', overall_accuracies[test_iter].item()
              )
        
        prev_task_data.append(data_dict_val)
        
        test_iter += 1
        
        '''
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
           
            torch.save(mlp.state_dict(), model_path ) 
        '''
        
    print("stop")
    print("stop")     
    print("stop")










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
   
   expr(model_params , data_params)



if __name__ == '__main__':
    
    
    model_config_path = os.path.join( ROOT, "configuration_files","cifar_100", "models", "bp", "0.json") 
    
    data_config_path = os.path.join( ROOT, "configuration_files","cifar_100", "data", "0.json")
    
    sys.exit( main ( ['-c1', model_config_path, '-c2', data_config_path ] ) )
 