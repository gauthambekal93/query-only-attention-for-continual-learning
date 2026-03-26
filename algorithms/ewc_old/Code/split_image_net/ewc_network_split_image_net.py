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
import numpy as np
import time 
import random


class EWC:
    def __init__(self, model, train_x, train_y, fisher_samples , mini_batch_size):
        """
        Computes diagonal Fisher and stores parameter means (theta*).
        - model: trained on the just-finished task (set to eval).
        - dataloader: data from the finished task (or a subsample).
        - fisher_samples: cap the number of samples used to estimate Fisher.
        """
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        # snapshot of parameters θ*
        self.means = {n: p.detach().clone() for n, p in self.params.items()}
        # estimate diagonal Fisher
        self.fisher = self.estimate_fisher(model, train_x, train_y, fisher_samples , mini_batch_size)

    @torch.no_grad()
    def zeros_like_params(self):
        return {n: torch.zeros_like(p, device=p.device) for n, p in self.params.items()}

    def estimate_fisher(self, model, train_x, train_y, fisher_samples , mini_batch_size):
         
         model.eval()
         
         fisher = self.zeros_like_params()
         
         seen = 0
         
         #for idx in range(0, len(train_x), mini_batch_size):
         
         for idx in range (0, fisher_samples, mini_batch_size):
             
             x = train_x[idx : idx + mini_batch_size]
             
             y = train_y[idx : idx + mini_batch_size].reshape(-1)
     
             logits = model.forward(x)
             # sample targets from model’s predictive distribution (for classification,
             # MAP targets via argmax is common; or use given labels y if you have them)
             # Using given labels y is typical in EWC papers.
             logp = F.log_softmax(logits, dim=1)
             # negative log-likelihood of correct class
             nll = F.nll_loss(logp, y.reshape(-1), reduction='mean')
             
             grads = torch.autograd.grad(nll, [p for p in model.parameters() if p.requires_grad], create_graph=False, retain_graph=False, allow_unused=True)

             with torch.no_grad():
                 i = 0
                 for name, p in model.named_parameters():
                     if not p.requires_grad:
                         continue
                     g = grads[i]
                     i += 1
                     if g is not None:
                         fisher[name] += (g.detach() ** 2) * mini_batch_size # weight by batch size
             seen += mini_batch_size
     
         # normalize by number of samples we actually used
         denom = max(1, seen)
         for name in fisher:
             fisher[name] /= denom
     
         return fisher
             

    def penalty(self, model):
        """
        EWC quadratic penalty for the current model relative to stored (means, fisher).
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                diff = p - self.means[n]
                loss = loss + (self.fisher[n] * diff.pow(2)).sum()
        return loss
    
    
    
class EWC_Net(nn.Module):
    
    def __init__(self, input_features= 49, num_features=2000, num_outputs=1, act_type='relu', opt ='adam', step_size = 0.001,
                 weight_decay=0,  momentum=0, beta_1=0.9, beta_2=0.999, loss='mse', task_datapoints = 10000, classes_per_task = 10, ewc_lambda = 1000):
            
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
            self.opt = optim.SGD(self.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.parameters(), lr=step_size)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.parameters(), lr=step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
            
        self.loss = loss
        
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]
        
        self.logits_1 = None
        
        self.activation_outputs = torch.ones( task_datapoints, self.fc1.out_features  )
        
        self.datapoint_count = 0
        
        self.weight_magnitude = []

        self.all_fisher_estimates = []
        
        self.ewc_lambda = ewc_lambda
        
    def forward(self, train_x):
        
        self.cnn_logits = self.cnn(train_x).flatten(1)
        
        self.logits_1 = F.relu(self.fc1( self.cnn_logits ))
        
        self.logits_2 =  F.relu(self.fc2(self.logits_1) ) 
        
        y_pred =  self.fc3(self.logits_2)  
        
        return y_pred
    

    def learn(self, train_x, train_y, train_y_one_hot, ewc_objects ):
        
        self.train()
        
        self.opt.zero_grad()
        
        y_pred = self.forward(train_x)
        
        ewc_pen = 0.0
        
        if ewc_objects:
            
            for e in ewc_objects:
                ewc_pen = ewc_pen + e.penalty(self)
            
            ewc_pen = (self.ewc_lambda / 2.0) * ewc_pen

        ce = self.loss_func(y_pred, train_y_one_hot)
        
        loss = ce + ewc_pen
        
        loss.backward()
        
        self.opt.step()
        
        episode_loss = loss
    
        accuracies = 100 * (y_pred.argmax(dim = 1)==train_y_one_hot.argmax(dim =1)).float().mean()
          
        return episode_loss, accuracies

    
    
    def test(self, data_dict_val ):
        
        test_x, test_y = data_dict_val["test_x"], data_dict_val["test_y"]
        
        y_pred = self.forward(test_x)
            
        accuracies = 100* (y_pred.argmax(dim = 1)==test_y.argmax(dim =1)).float().mean()
            
        return accuracies
        

        
        
        
        
        