# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 21:16:32 2025

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:46:47 2025

@author: gauthambekal93
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch


class MAML(nn.Module):
    

    def __init__(self, input_features , num_features, num_outputs, num_hidden_layers, act_type, opt, inner_step_size, outer_step_size , weight_decay, momentum, beta_1, beta_2, 
                 loss, task_datapoints , classes_per_task ,inner_loop_count, num_of_task_sampled):
        
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3,  32,  5, padding=1), nn.ReLU(), nn.MaxPool2d(2),  
            nn.Conv2d(32, 64,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(input_features , num_features)
        
        self.fc2 = nn.Linear(num_features, num_features)
        
        self.fc3 = nn.Linear(num_features, num_outputs)
        
        self.classes_per_task = classes_per_task
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        if opt == 'sgd':
            self.opt = optim.SGD(self.parameters(), lr=outer_step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.parameters(), lr=outer_step_size)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.parameters(), lr=outer_step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
            
        self.loss = loss
        
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]
        
        self.logits_1 = None
        
        self.activation_outputs = torch.ones( task_datapoints, self.fc1.out_features  )
        
        self.datapoint_count = 0
        
        self.weight_magnitude = []
       
        self.no_of_task_sampled = num_of_task_sampled
        
        self.inner_loop_count = inner_loop_count
        
        self.inner_step_size = inner_step_size
        
    
    def get_theta_prime_loss(self, theta_prime, x, y):
        
        """HERE WE ALSO NEED TO RETUN THE ACCURACY """
        
        w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6 = theta_prime
        
        logits_1 = F.max_pool2d ( F.relu( torch.nn.functional.conv2d(x, w1, b1, stride=1, padding=1) ), kernel_size=2, stride=2  )
        
        logits_2 = F.max_pool2d (  F.relu( torch.nn.functional.conv2d(logits_1, w2, b2, stride=1, padding=1 )),  kernel_size=2, stride=2  )
        
        logits_3 = F.max_pool2d (  F.relu( torch.nn.functional.conv2d(logits_2, w3, b3, stride=1, padding=1)),  kernel_size=2, stride=2  )
        
        logits_3 = logits_3.flatten(1)
        
        logits_4 = F.relu( torch.nn.functional.linear(logits_3, w4, b4))
        
        logits_5 = F.relu( torch.nn.functional.linear(logits_4, w5, b5))
        
        y_pred = torch.nn.functional.linear(logits_5, w6, b6)
    
        theta_prime_loss = self.loss_func(y_pred, y)
        
        accuracies = 100 * ((y_pred.argmax(dim = 1) == y.argmax(dim = 1) ).float().sum() / len(y) )
       
        return theta_prime_loss, accuracies
    
    
    
    def update_theta_prime(self, theta, x, y ):
        
        theta_prime = [p.clone().requires_grad_(True) for p in theta]
        
        for _ in range(self.inner_loop_count):
            
            loss, _ =  self.get_theta_prime_loss( theta_prime, x, y)
            
            grads = torch.autograd.grad(loss, theta_prime, create_graph=True)  
            
            theta_prime = [ theta_prime[i] - self.inner_step_size * grads[i] for i in range(len(theta_prime)) ]
        
        return theta_prime
        
        
    
    def get_meta_gradients(self, theta_prime_loss, theta):
        
        """meta_params used to obtain prediction and hence loss. Then diffrentiate loss wrt meta_params. 
        The create_graph=True for calculating the hessian. So that meta_grads have the computational graph"""
        meta_grads = torch.autograd.grad(theta_prime_loss, theta, create_graph=True)
        
        return meta_grads
    
    
    def learn(self, support_dict, query_dict ):
       
        theta = list(self.parameters())
        
        theta_primes = []
        
        task_indices = list(support_dict.keys())#[:self.no_of_task_sampled]
        
        for i, index_val in enumerate(task_indices):
            
            support_x, support_y = support_dict[index_val]["support_x"], support_dict[index_val]["support_y"]
            
            theta_prime = self.update_theta_prime( theta, support_x, support_y )
            
            theta_primes.append(theta_prime)
        
        theta_prime_loss, accuracies = [], 0
        
        for i, index_val in enumerate(task_indices):
        
            theta_prime = theta_primes[i]
            
            query_x, query_y = query_dict[index_val]['query_x'], query_dict[i]['query_y']
            
            loss, accuracy = self.get_theta_prime_loss(theta_prime, query_x, query_y) 
            
            theta_prime_loss.append(loss )
            
            accuracies += accuracy
        
        accuracies = accuracies/ len(theta_primes)
        
        theta_prime_loss = torch.mean(torch.stack(theta_prime_loss))
        
        meta_grads = self.get_meta_gradients(theta_prime_loss, theta)
        """We need to update grad paramter of outer_prams since we are using pytorch built in optimizer like adam. """
        
        for p, g in zip(theta, meta_grads): 
             """ .deatch() here is not necessary since diffrentiation is already done and we are zeroing 
             the gradients after step. so this for best practice and safety.  """
             
             p.grad = g.detach()  
     
        self.opt.step()
         
        self.opt.zero_grad()
        
        return theta_prime_loss.item(), accuracies
    

    
    
    def test(self, data_dict_val ):
        
        support_x, support_y, queries_x, queries_y = data_dict_val["support_x"], data_dict_val["support_y"], data_dict_val["query_x"], data_dict_val["query_y"]
        
        theta = list(self.parameters())
        
        theta_prime = self.update_theta_prime( theta, support_x, support_y )
        
        with torch.no_grad():
            theta_prime_loss, accuracies = self.get_theta_prime_loss(theta_prime, queries_x, queries_y)
            
        return accuracies
    
   
    
        
        
 
        
        
        
        
        
        
        
        