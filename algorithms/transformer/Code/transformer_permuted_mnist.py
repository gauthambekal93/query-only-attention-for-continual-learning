# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:46:23 2025

@author: gauthambekal93
"""

import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
from torch import optim
import time


class Transformers(nn.Module):
      
    def __init__(self, feature_size= 49, num_outputs=1, num_hidden_layers=2, act_type='relu', opt ='adam', step_size = 0.001,
                  weight_decay=0,  momentum=0, beta_1=0.9, beta_2=0.999, loss='mse', task_datapoints = 10000, classes_per_task = 10):
        
        super().__init__()
        
        self.feature_size = feature_size
        self.query_1 = nn.Linear(feature_size, feature_size)
        self.key_1 = nn.Linear(feature_size, feature_size)
        self.value_1 = nn.Linear(feature_size, feature_size)
        
        self.query_2 = nn.Linear(feature_size, feature_size)
        self.key_2 = nn.Linear(feature_size, feature_size)
        self.value_2 = nn.Linear(feature_size, feature_size)
        
        self.query_3 = nn.Linear(feature_size, feature_size)
        self.key_3 = nn.Linear(feature_size, feature_size)
        self.value_3 = nn.Linear(feature_size, feature_size)
        
        self.query_4 = nn.Linear(feature_size, feature_size)
        self.key_4 = nn.Linear(feature_size, feature_size)
        self.value_4 = nn.Linear(feature_size, feature_size)
        
        self.query_5 = nn.Linear(feature_size, feature_size)  #HAVE to possibly change output dimensions !!!
        self.key_5 = nn.Linear(feature_size, feature_size)
        self.value_5 = nn.Linear(feature_size, feature_size)
        
        self.query_6 = nn.Linear(feature_size, feature_size)
        self.key_6 = nn.Linear(feature_size, feature_size)
        self.value_6 = nn.Linear(feature_size, feature_size)
        
        self.query_7 = nn.Linear(feature_size, feature_size)
        self.key_7 = nn.Linear(feature_size, feature_size)
        self.value_7 = nn.Linear(feature_size, feature_size)
        
    
        self.query_8 = nn.Linear(feature_size, feature_size)
        self.key_8 = nn.Linear(feature_size, feature_size)
        self.value_8 = nn.Linear(feature_size, feature_size)
        
        self.query_9 = nn.Linear(feature_size, feature_size)
        self.key_9 = nn.Linear(feature_size, feature_size)
        self.value_9 = nn.Linear(feature_size, feature_size)
        
        self.query_10 = nn.Linear(feature_size, feature_size)  #HAVE to possibly change output dimensions !!!
        self.key_10 = nn.Linear(feature_size, feature_size)
        self.value_10 = nn.Linear(feature_size, feature_size)
        
        
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
            #self.opt = optim.Adam(self.parameters(), lr=step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.parameters(), lr=step_size, betas=(beta_1, beta_2), weight_decay=weight_decay)
        
        self.loss = loss
        
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]
        
        self.weight_magnitude = []
        
        
    def forward(self, Z):
        
        "----Layer 1----"
        query_output_1 = self.query_1(Z)
        
        key_output_1 = self.key_1(Z)
        
        value_output_1 = self.value_1(Z)
        
        attention_matrix = F.softmax( torch.matmul(query_output_1, key_output_1.transpose(-2, -1) ) / math.sqrt( self.key_1.in_features ) , dim = -1 )
        
        output = torch.matmul( attention_matrix, value_output_1)
        
        "----Residual Connection---"
        layer1_output = Z + output
        
        
        "----Layer 2----"
        query_output_2 = self.query_2(layer1_output)
        
        key_output_2 = self.key_2(layer1_output)
        
        value_output_2 = self.value_2(layer1_output)
        
        attention_matrix = F.softmax( torch.matmul(query_output_2, key_output_2.transpose(-2, -1) ) / math.sqrt( self.key_2.in_features ) , dim = -1 )
        
        output = torch.matmul( attention_matrix, value_output_2)
       
        "----Residual Connection---"
        layer2_output =  layer1_output + output
        
    
        
        query_output_3 = self.query_3(layer2_output)
        
        key_output_3 = self.key_3(layer2_output)
        
        value_output_3 = self.value_3(layer2_output)
        
        attention_matrix = F.softmax( torch.matmul(query_output_3, key_output_3.transpose(-2, -1) ) / math.sqrt( self.key_3.in_features ) , dim = -1 )
        
        output = torch.matmul( attention_matrix, value_output_3)
        
        "----Residual Connection---"
        layer3_output =  layer2_output + output
        
        
    
        query_output_4 = self.query_4(layer3_output)
        
        key_output_4 = self.key_4(layer3_output)
        
        value_output_4 = self.value_4(layer3_output)
        
        attention_matrix = F.softmax( torch.matmul(query_output_4, key_output_4.transpose(-2, -1) ) / math.sqrt( self.key_4.in_features ) , dim = -1 )
        
        output = torch.matmul( attention_matrix, value_output_4)
        
        "----Residual Connection---"
        layer4_output =  layer3_output + output
        

    
        query_output_5 = self.query_5(layer4_output)
        
        key_output_5 = self.key_5(layer4_output)
        
        value_output_5 = self.value_5(layer4_output)
        
        attention_matrix = F.softmax( torch.matmul(query_output_5, key_output_5.transpose(-2, -1) ) / math.sqrt( self.key_5.in_features ) , dim = -1 )
        
        output = torch.matmul( attention_matrix, value_output_5)
        
        "----Residual Connection---"
        layer5_output =  layer4_output + output
        
        
        query_output_6 = self.query_6(layer5_output)
        
        key_output_6 = self.key_6(layer5_output)
        
        value_output_6 = self.value_6(layer5_output)
        
        attention_matrix = F.softmax( torch.matmul(query_output_6, key_output_6.transpose(-2, -1) ) / math.sqrt( self.key_6.in_features ) , dim = -1 )
        
        output = torch.matmul( attention_matrix, value_output_6)
        
        "----Residual Connection---"
        layer6_output = layer5_output + output   #there is error here!
        
        
        "----Layer 2----"
        query_output_7 = self.query_7(layer6_output)
        
        key_output_7 = self.key_7(layer6_output)
        
        value_output_7 = self.value_7(layer6_output)
        
        attention_matrix = F.softmax( torch.matmul(query_output_7, key_output_7.transpose(-2, -1) ) / math.sqrt( self.key_7.in_features ) , dim = -1 )
        
        output = torch.matmul( attention_matrix, value_output_7)
       
        "----Residual Connection---"
        layer7_output =  layer6_output + output
        
    
        
        query_output_8 = self.query_8(layer7_output)
        
        key_output_8 = self.key_8(layer7_output)
        
        value_output_8 = self.value_8(layer7_output)
        
        attention_matrix = F.softmax( torch.matmul(query_output_8, key_output_8.transpose(-2, -1) ) / math.sqrt( self.key_8.in_features ) , dim = -1 )
        
        output = torch.matmul( attention_matrix, value_output_8)
        
        "----Residual Connection---"
        layer8_output =  layer7_output + output
        
        
    
        query_output_9 = self.query_9(layer8_output)
        
        key_output_9 = self.key_9(layer8_output)
        
        value_output_9 = self.value_9(layer8_output)
        
        attention_matrix = F.softmax( torch.matmul(query_output_9, key_output_9.transpose(-2, -1) ) / math.sqrt( self.key_9.in_features ) , dim = -1 )
        
        output = torch.matmul( attention_matrix, value_output_9)
        
        "----Residual Connection---"
        layer9_output =  layer8_output + output
        

    
        query_output_10 = self.query_10(layer9_output)
        
        key_output_10 = self.key_10(layer9_output)
        
        value_output_10 = self.value_10(layer9_output)
        
        attention_matrix = F.softmax( torch.matmul(query_output_10, key_output_10.transpose(-2, -1) ) / math.sqrt( self.key_10.in_features ) , dim = -1 )
        
        output = torch.matmul( attention_matrix, value_output_10)
        
        
        "----Residual Connection---"
        layer10_output =  layer9_output + output
        
        
        if layer10_output.ndim  == 3:
            y_pred = layer10_output[:, -1:, - self.classes_per_task :].squeeze(1)  #this will have to be updated since output will be 10 d !!!
        
        if layer10_output.ndim == 2:
            y_pred = layer10_output[-1:, - self.classes_per_task :]
        '''
        
        if layer7_output.ndim  == 3:
            y_pred = layer7_output[:, -1:, - self.classes_per_task :].squeeze(1)  #this will have to be updated since output will be 10 d !!!
        
        if layer7_output.ndim == 2:
            y_pred = layer7_output[-1:, - self.classes_per_task :]
        '''
        
        return y_pred
    
    
    def learn(self, X, Y):
        """
        Learn using one step of gradient-descent
        :param x: input
        :param target: desired output
        :return: loss
        """

        self.opt.zero_grad()
        
        y_pred = self.forward(X)
        
        loss = self.loss_func(y_pred, Y)

        loss.backward()
        
        self.opt.step()
        
        episode_loss = loss.item()
        
        accuracies = 100 *  (y_pred.argmax(dim = 1)==Y.argmax(dim = 1)).float().sum()
        
        
        return episode_loss, accuracies
        
    
    def test(self, X, Y ):
         
         episode_loss  = []
         
         X = torch.stack(X, dim = 0)
         
         Y = torch.stack(Y, dim = 0)  
        
         y_pred = self.forward( X )
         
         loss = self.loss_func(y_pred, Y)
         
         episode_loss.append(loss.item())
         
         accuracies = 100* ( (y_pred.argmax(dim = 1)==Y.argmax(dim =1 )).sum().float() ) / Y.shape[0]
         
         return episode_loss, accuracies

         '''
         episode_loss, accuracies = [], 0
        
         for z, y in zip(Z, Y):
            
            y_pred= self.forward( z )
        
            y = y.expand(y_pred.shape)
            
            episode_loss.append( self.loss_func(y_pred,  y ) )
       
            accuracies += (y_pred.argmax()==y.argmax()).float()
     
         episode_loss = np.mean( episode_loss )
   
         accuracies = 100* (accuracies / len(Z))  #was self.classes_per_task

         return episode_loss, accuracies
         '''           
    
    
    def calculate_hessian(self,Z, Y):
        
        params = list(self.query_7.parameters() )
        
        loss = []
        
        """we only take a single datapoint for hessian calculation to minimize the compute  """
        for z, y in zip(Z, Y):
            
            y_pred = self.forward( z )
        
            loss.append( self.loss_func(y_pred,  y.expand( y_pred.shape) ) )
            
            break
            
        
        loss = torch.stack(loss).mean()
        
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # last row of weight and  last element of bias
        grads_flat = torch.cat([ grads[0][-1, :].reshape(-1), grads[1][-1:].reshape(-1) ])
        
        # Compute Hessian (final layer only)
        num_params = grads_flat.numel()
        
        H = torch.zeros((num_params, num_params))
        
        for i in range(num_params):
            
            second_grads = torch.autograd.grad(grads_flat[i], params, retain_graph=True)
        
            H[i] = torch.cat([ second_grads[0][-1, :].reshape(-1), second_grads[1][-1:].reshape(-1) ]).detach()
    
        
        try:
            epsilon = 1e-6
            H_stable = H + epsilon * torch.eye(H.shape[0])
            eigenvalues = torch.linalg.eigvalsh(H_stable)
        except:
            print("stop")
            print("stop")
            print("stop")
            
        eigenvalues = torch.clamp(eigenvalues, min=1e-12)  # Prevent log(0)
        
        p = eigenvalues / eigenvalues.sum()
        
        entropy = -torch.sum(p * torch.log(p))
        
        effective_rank = torch.exp(entropy)
        
        return effective_rank
    
    
    def calculate_weight_magnitude(self):
           total = 0
           count = 0
           for name, param in self.named_parameters():
                #print("Param name ", name)
                total += param.abs().sum().item()
                count += param.numel()
           return total / count
           
        