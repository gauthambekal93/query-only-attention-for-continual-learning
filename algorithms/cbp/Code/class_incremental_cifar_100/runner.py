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
      
    
    def set_lr(self, epoch, train_context):
            """ Changes the learning rate of the optimizer according to the current epoch of the task """
            current_stepsize = None
            
            if epoch  == 0:
                current_stepsize = train_context.step_size

            elif epoch == 60:
                current_stepsize = round(train_context.step_size * 0.2, 5)
            
            elif epoch == 120:
                current_stepsize = round(train_context.step_size * (0.2 ** 2), 5)
            
            elif epoch == 160:
                current_stepsize = round(train_context.step_size * (0.2 ** 3), 5)

            if current_stepsize is not None:
                for g in train_context.optim.param_groups:
                    g['lr'] = current_stepsize
                    
                
    def evaluvate_network(self, epoch, train_context, data_manager_obj, checkpoint_obj):
        
        train_context.net.eval()
        
        avg_loss = 0.0
        avg_acc = 0.0
        num_test_batches = 0
        
        start = time.perf_counter()
        with torch.no_grad():
            test_ids = torch.arange( len(data_manager_obj.task_test_y) )
            
            no_of_batches = len(data_manager_obj.task_test_y) // self.test_batch_size
            
            for batch_no in range( no_of_batches ): 
                
                start_id , end_id = self.test_batch_size  * batch_no,  self.test_batch_size  * (batch_no +1)
                
                batch_ids = test_ids[start_id: end_id ]
                
                batch_x, batch_y = data_manager_obj.task_test_x[batch_ids], data_manager_obj.task_test_y[batch_ids]
                
                predictions = train_context.net.forward(batch_x )[:, data_manager_obj.selected_classes]
                
                avg_loss += train_context.loss(predictions, batch_y)
                
                avg_acc += torch.mean((predictions.argmax(axis=1) == batch_y.argmax(axis=1)).to(torch.float32))
                
                num_test_batches += 1
                
        print("t2" ,time.perf_counter() - start)         
        
        checkpoint_obj.summarize_test(avg_loss / num_test_batches, avg_acc / num_test_batches, 
                                      data_manager_obj.current_task_id, data_manager_obj.current_num_classes )
    
        
 
    def train(self, train_context, data_manager_obj, checkpoint_obj):

        """train model """
        start = time.perf_counter()
        
        train_context.net.train()
        
        for epoch in tqdm( range(self.epochs_per_task)):
            
            self.set_lr(epoch, train_context)
            
            rand_idx = torch.randperm(len(data_manager_obj.task_train_y)) 
            
            no_of_batches = len(rand_idx) // self.train_batch_size
            
            for batch_no in range( no_of_batches ): 
                
                start = time.perf_counter()
                
                start_id , end_id = self.train_batch_size  * batch_no,  self.train_batch_size  * (batch_no +1)
                
                batch_ids = rand_idx[start_id: end_id ]
                            
                batch_x, batch_y = data_manager_obj.task_train_x[batch_ids], data_manager_obj.task_train_y[batch_ids]
                
                batch_x = data_manager_obj.augment_batch(batch_x)
                
                for param in train_context.net.parameters(): 
                    param.grad = None   # apparently faster than optim.zero_grad()
                
                train_context.current_features = []
                
                predictions = train_context.net.forward( batch_x , train_context.current_features)[:, data_manager_obj.selected_classes] #t2 0.34693420003168285
                
                current_reg_loss = train_context.loss(predictions, batch_y)
                
                current_reg_loss.backward()
                
                train_context.optim.step()
                
                train_context.resgnt.gen_and_test(train_context.current_features)
                
                checkpoint_obj.running_accuracy += torch.mean((predictions.argmax(axis=1) == batch_y.argmax(axis=1)).to(torch.float32)).detach()
                checkpoint_obj.running_loss += current_reg_loss.detach()
                
                
                """save checkpoints """
                if (batch_no + 1) % checkpoint_obj.running_avg_window == 0  :
                    
                    checkpoint_obj.summarize_train()

        """obtain performance """
        self.evaluvate_network(epoch, train_context, data_manager_obj, checkpoint_obj)
        
        print("t1", time.perf_counter() - start   ) 
        
        checkpoint_obj.save_experiment_checkpoint(train_context)
        
        
        
    def run(self, train_context, data_manager_obj, checkpoint_obj):
        
        while data_manager_obj.current_task_id < data_manager_obj.total_tasks:
            
            data_manager_obj.create_task_data()
            
            self.train( train_context, data_manager_obj, checkpoint_obj)
            
            data_manager_obj.current_task_id += 1
            
            data_manager_obj.current_num_classes += data_manager_obj.class_increase_per_task
            
'''             
for i, group in enumerate(train_context.optim.param_groups):
    print(f"Param group {i} LR = {group['lr']}")         
'''    
        
        