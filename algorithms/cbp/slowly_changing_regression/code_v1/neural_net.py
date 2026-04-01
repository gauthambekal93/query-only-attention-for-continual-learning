# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 11:17:31 2025

@author: gauthambekal93
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim

import GnT
import AdamGnT
    


class feed_forward_nn(nn.Module):
    def __init__(self, input_size, num_features, num_outputs , opt, replacement_rate , decay_rate, maturity_threshold, util_type, device, loss_func, init, accumulate):
        super(feed_forward_nn, self).__init__()
        self.num_inputs = input_size
        self.num_features = num_features
        self.num_outputs = num_outputs
        
        self.fc1 = nn.Linear(input_size, num_features)
        
        self.fc2 = nn.Linear(num_features, num_outputs)
    
        
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)
        

        self.gnt = None
        self.gnt = GnT(
            net=self.net.layers, #neural net layers
            hidden_activation=self.net.act_type, #activation
            opt = opt,                     #optimization
            replacement_rate=replacement_rate,  #rate at which neurons are reinitialized
            decay_rate=decay_rate,
            maturity_threshold=maturity_threshold,   #when a neuron is reinitialized and has utility zero we dont want it to get reinitialized again in next time step
            util_type=util_type,
            device=device,
            loss_func=loss_func,
            init=init,             #kaiming
            accumulate=accumulate,  #boolean value
        )
        
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
       
        output, features = self.net.predict(x=x) # output is the output of nn, features is output after the final hidden layer afer activation
        loss = self.loss_func(output, target)
        self.previous_features = features

        # do the backward pass and take a gradient step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # take a generate-and-test step
        self.opt.zero_grad()
        if type(self.gnt) is GnT:
            self.gnt.gen_and_test(features=self.previous_features)

        if self.loss_func == F.cross_entropy:
            return loss.detach(), output.detach()
  
            
        return loss.detach()
    
    
    