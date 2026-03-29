# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 11:17:31 2025

@author: gauthambekal93
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim


    


class feed_forward_nn(nn.Module):
    def __init__(self, input_size, num_features,num_outputs, hidden_activation  ):
        super(feed_forward_nn, self).__init__()
        self.num_inputs = input_size
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.act_type = hidden_activation

        # define the hidden activation
        self.hidden_activation = {'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'selu': nn.SELU,
                                  'swish': nn.SiLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}[self.act_type]

        # define the architecture
        self.layers = nn.ModuleList()  #instead of say self.fc1 = nn.Linear(input_size , num_features) we are storing in a ModuleList
        self.layers.append(nn.Linear(input_size, num_features))  
        self.layers.append(self.hidden_activation())
        self.layers.append(nn.Linear(num_features, num_outputs))

        # initialize the input weights
        self.layers[0].bias.data.fill_(0.0)
        if hidden_activation in ['sigmoid', 'relu', 'tanh', 'leaky_relu']:
            nn.init.kaiming_uniform_(self.layers[0].weight, nonlinearity=hidden_activation)
        elif hidden_activation in ['swish', 'elu']:
            nn.init.kaiming_uniform_(self.layers[0].weight, nonlinearity='relu')
        # initialize the output weights
        nn.init.kaiming_uniform_(self.layers[-1].weight, nonlinearity='linear') #usingKaiming (He) initialization to initialize the weights of the last layer of a neural network
        self.layers[-1].bias.data.fill_(0.0) #It sets all the bias values of the last layer to zero.


    def forward(self, x):
        
        features = self.layers[1](self.layers[0](x))
        output = self.layers[-1](features)
        return output
    
    
    
    