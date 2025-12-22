# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:15:04 2025

@author: gauthambekal93
"""


import torch
from tqdm import tqdm
import time

class Runner:
    
    def __init__(self,  data_manager_obj, num_epochs, train_batch_size, test_batch_size):
        
        self.num_epochs = num_epochs
        self.epochs_per_task = int( self.num_epochs / data_manager_obj.total_tasks ) #epochs_per_task
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        
        
        #self.current_num_classes = current_num_classes
        #self.class_increase_per_task = class_increase_per_task
        #self.current_task_id = 0
        #self.total_tasks =  total_tasks
      
    
    def evaluvate_network(self, epoch, train_context, data_manager_obj, checkpoint_obj):
        
        avg_loss = 0.0
        avg_acc = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            test_ids = torch.range( len(data_manager_obj.task_test_y) )
            
            for batch_no in range( len(test_ids) ):
                
                batch_ids = test_ids[batch_no: batch_no + self.test_batch_size ]
            
                batch_x, batch_y = data_manager_obj.task_test_x[batch_ids].to(train_context.device), data_manager_obj.task_test_y[batch_ids].to(train_context.device)
                
                predictions = train_context.net.forward(batch_x )[:, self.all_classes[:data_manager_obj.current_num_classes]]
                
                avg_loss += train_context.loss(predictions, batch_y)
                
                avg_acc += torch.mean((predictions.argmax(axis=1) == batch_y.argmax(axis=1)).to(torch.float32))
                
                num_test_batches += 1
            
        checkpoint_obj.summarize_test(avg_loss / num_test_batches, avg_acc / num_test_batches, epoch )
         
        
 
    def train(self, train_context, data_manager_obj, checkpoint_obj):

        """train model """
        for epoch in tqdm( range(self.epochs_per_task)):
            
            #selected_classes = torch.tensor(data_manager_obj.label_ids_flattened [:data_manager_obj.current_num_classes] , device=train_context.device, dtype=torch.long)
            
            train_context.net.train()
            
            rand_idx = torch.randperm(len(data_manager_obj.task_train_y)) 
            
            for batch_no in range( 0, len(rand_idx), self.train_batch_size ):
                
                start = time.perf_counter()
                
                batch_ids = rand_idx[batch_no: batch_no + self.train_batch_size ]
            
                #batch_x, batch_y = data_manager_obj.task_train_x[batch_ids].to(train_context.device), data_manager_obj.task_train_y[batch_ids].to(train_context.device)
                
                batch_x, batch_y = data_manager_obj.task_train_x[batch_ids], data_manager_obj.task_train_y[batch_ids]
                
                print("t1" ,time.perf_counter() - start)   #t1 0.000588799943216145  wass 0.014849600032903254
                
                for param in train_context.net.parameters(): 
                    param.grad = None   # apparently faster than optim.zero_grad()
                
                start = time.perf_counter()
                predictions = train_context.net.forward(batch_x )[:, data_manager_obj.selected_classes] #t2 0.34693420003168285
                print("t2" ,time.perf_counter() - start)
                
                start = time.perf_counter()
                current_reg_loss = train_context.loss(predictions, batch_y)
                print("t3" ,time.perf_counter() - start)
                
                start = time.perf_counter()
                current_reg_loss.backward()
                print("t4" ,time.perf_counter() - start)
               
                start = time.perf_counter()
                train_context.optim.step()
                print("t5" ,time.perf_counter() - start)
                
                start = time.perf_counter()
                checkpoint_obj.running_accuracy += torch.mean((predictions.argmax(axis=1) == batch_y.argmax(axis=1)).to(torch.float32)).detach()
                checkpoint_obj.running_loss += current_reg_loss.detach()
                print("t6" ,time.perf_counter() - start)
                
                
                """save checkpoints """
                if (batch_no + 1) % checkpoint_obj.running_avg_window == 0:
                    start = time.perf_counter()
                    checkpoint_obj.summarize_train()
                    print("t7" ,time.perf_counter() - start)
                    
            train_context.net.eval()
            """obtain performance """
        self.evaluvate_network(epoch, train_context, data_manager_obj, checkpoint_obj)
        
     
        
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        while data_manager_obj.current_task_id < data_manager_obj.total_tasks:
            
            data_manager_obj.create_task_data()
            
            self.train( train_context, data_manager_obj, checkpoint_obj)
            
            data_manager_obj.current_task_id += 1
            
            data_manager_obj.current_num_classes += data_manager_obj.class_increase_per_task
            
        checkpoint_obj.save_experiment_checkpoint(train_context)     
           
    
        
        