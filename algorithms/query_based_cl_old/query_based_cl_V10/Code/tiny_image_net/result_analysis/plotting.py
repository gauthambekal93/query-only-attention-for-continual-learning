# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 10:16:44 2026

@author: gauthambekal93
"""

import matplotlib.pyplot as plt

def plot_data(data, metric, plot_type):
    plt.figure(figsize=(6,4))
    plt.plot(data, marker='o')
    plt.xlabel("Task")
    plt.ylabel(metric)
    plt.title(plot_type)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    
 
train_loss =  [ loss for loss in self.results_dict['train_loss'].values()]    

global_test_acc =  [ loss for loss in  self.results_dict["global_test_acc"].values()]    

plot_data(train_loss, metric = "loss", plot_type = "task vs train loss")


plot_data(global_test_acc, metric = "global test accuracy", plot_type = "task vs global test accuracy")
