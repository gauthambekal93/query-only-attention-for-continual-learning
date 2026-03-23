# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:26:21 2026

@author: gauthambekal93
"""



import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent  # go up two levels, adjust as needed


# === CONFIG ===================================================================
# Set this to your repo root that contains results/permuted_mnist/<model>/<run>/<lr>/output.pkl


# Plot markers every N points (line still connects all points)
PLOT_EVERY = 50
#average_over = 50
# Number of tasks to truncate to for the long-running methods
#NUM_TASKS_LONG = 6000  # will be clipped to available length per series

# ==============================================================================
def _load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

'''
def calculate_curve(model_type, run_numbers, lr_index, key, num_tasks=None, average_over = 0, sub_model_type = None):
    """
    Loads output.pkl from multiple seeds (run_numbers) and returns mean curve over runs.
    key: 'forward_accuracies' | 'backward_accuracies' | 'forward_effective_ranks' | ...
    num_tasks: optional truncate length
    """
    runs = []
    for rn in run_numbers:
        if sub_model_type is None:
            p = os.path.join(project_root, "results", "cifar_100", model_type, rn, lr_index, "result.pkl")
        else:
            p = os.path.join(project_root, "results", "cifar_100", model_type, sub_model_type,  rn, lr_index, "result.pkl")
        out = _load_pickle(p)
        arr = np.array([v for k, v in out[key].items()])   [: num_tasks] 
            
            
        runs.append(arr)
        
    runs = np.stack(runs, axis=1)  # [T, R]
    mean_curve = runs.mean(axis=1)  # [T]
    std  = runs.std(axis = 1)  
    
    
    """Block avg """
    mean_curve = np.array( [np.mean(mean_curve[i: i+ average_over]) for i in range (0, len (mean_curve), average_over )])
    std = np.array([np.mean(std[i: i+ average_over]) for i in range (0, len (std), average_over )])
    """Running avg """
    #mean_curve = np.array( [np.mean(mean_curve[i:i+average_over]) for i in range(0, len(mean_curve), average_over)])
    #std = np.array( [np.mean(std[i:i+average_over]) for i in range(0, len(std), average_over)])
            
    
    return mean_curve, std
'''

def calculate_curve( base_path, run_numbers, lr_index, key, num_tasks=None, average_over = 0):
    """
    Loads output.pkl from multiple seeds (run_numbers) and returns mean curve over runs.
    key: 'forward_accuracies' | 'backward_accuracies' | 'forward_effective_ranks' | ...
    num_tasks: optional truncate length
    """
    runs = []
    for rn in run_numbers:
        p = os.path.join(base_path, rn, lr_index, "result.pkl" )
        out = _load_pickle(p)
        arr = np.array([v for k, v in out[key].items()])   [: num_tasks] 
        runs.append(arr)
        
    runs = np.stack(runs, axis=1)  # [T, R]
    mean_curve = runs.mean(axis=1)  # [T]
    std  = runs.std(axis = 1)  
    
    
    """Block avg """
    mean_curve = np.array( [np.mean(mean_curve[i: i+ average_over]) for i in range (0, len (mean_curve), average_over )])
    std = np.array([np.mean(std[i: i+ average_over]) for i in range (0, len (std), average_over )])
    """Running avg """
    #mean_curve = np.array( [np.mean(mean_curve[i:i+average_over]) for i in range(0, len(mean_curve), average_over)])
    #std = np.array( [np.mean(std[i:i+average_over]) for i in range(0, len(std), average_over)])
            
    
    return mean_curve, std


def plot_graph(series_dict, title, ylabel="Accuracy", hline_at=None, vline_at=None):
    """
    series_dict: {label: y | label: (y, {style})}
      style keys: color, marker, linewidth, alpha, plot_every, linestyle
    """
    plt.figure(figsize=(8, 4))
    for lbl, payload in series_dict.items():
        if isinstance(payload, tuple) and len(payload) == 3 and isinstance(payload[2], dict):
            y, std, style = payload
        else:
            y, style = payload, {}
        
        
        y = np.asarray(y, dtype=float)
        x = np.arange(len(y))
    
        line_width = 1.3
        plt.plot(
            x, y,
            color=style.get("color", None),
            linewidth=style.get("linewidth", line_width),
            alpha=style.get("alpha", 0.95),
            label=lbl,
            marker=style.get("marker", 'o'),
            markersize=2,
            markevery=max(1, style.get("plot_every", PLOT_EVERY)),
            linestyle=style.get("linestyle", '-'),
        )
        
        x = np.arange(len(y))
        plt.fill_between(
            x,
            y - std,
            y + std,
            color=style.get("color", None),
            alpha=0.1
        )
        
        
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    if hline_at is not None:
        plt.axhline(hline_at, linewidth=1, alpha=0.5)
    if vline_at is not None:
        plt.axvline(vline_at, linewidth=1, alpha=0.5)
    plt.xlabel("Steps", fontsize = 13)
    plt.ylabel(ylabel, fontsize = 13)
    plt.title(title, fontsize=14)
    plt.grid(True, linewidth=0.3, alpha=0.5)
    plt.legend(ncol=3, fontsize=11, frameon=True,  loc="lower center", bbox_to_anchor=(0.65, 0.01) )
    plt.tight_layout()
    plt.show()

#def plot_rank_graph(series_dict, title):
#    plot_graph(series_dict, title=title, ylabel="Effective rank")

# === LOAD CURVES ==============================================================

# Common settings




""" ===============Experience replay with reservoir============"""



lr_index, runs, NUM_TASKS, average_over = "0" , ["0"] , 50000, 100
base_path = os.path.join(project_root, "results", "cifar_100", "AUGMENTATION", "experience_replay", "reservoir_replay")

fwd_er_replay, fwd_er_std  = calculate_curve(base_path, runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)
prequential_er_replay, prequential_er_std  = calculate_curve(base_path, runs, lr_index, "prequential_accuracy",  NUM_TASKS, average_over)
bwd_er_replay, bwd_er_std  = calculate_curve(base_path, runs, lr_index, "backward_accuracy",  NUM_TASKS, average_over)



plot_graph({ 
            "Forward Accurcy":  (fwd_er_replay, fwd_er_std, {"color":"red", "linestyle": "-",  "marker": "d"}),
            "Prequential Accuracy":  (prequential_er_replay, prequential_er_std, {"color":"green", "linestyle": "-",  "marker": "s"}),
            "Backward Accurcy":  (bwd_er_replay, bwd_er_std, {"color":"blue", "linestyle": "-",  "marker": "x"}),
            },
             title="Augmented CIFAR 100 - Experience replay Reservoir Buffer - 2000",)



"""==============Query based cl with balanced task rbuffer==================== """

lr_index, runs, NUM_TASKS, average_over = "0" , ["0"] , 50000, 10
base_path = os.path.join(project_root, "results", "cifar_100", "AUGMENTATION", "query_based_cl", "task_balanced_replay")

fwd_query_based_acc_4, fwd_query_based_std_4  = calculate_curve(base_path, runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)
prequential_query_based_acc_4, prequential_query_based_std_4  = calculate_curve(base_path, runs, lr_index, "prequential_accuracy",  NUM_TASKS, average_over)
bwd_query_based_acc_4, bwd_query_based_std_4  = calculate_curve(base_path, runs, lr_index, "backward_accuracy",  NUM_TASKS, average_over)


plot_graph({ 
            "Forward Accurcy":  (fwd_query_based_acc_4, fwd_query_based_std_4, {"color":"red", "linestyle": "-",  "marker": "d"}),
            "Prequential Accuracy":  (prequential_query_based_acc_4, prequential_query_based_std_4, {"color":"green", "linestyle": "-",  "marker": "s"}),
            #"Backward Accurcy":  (bwd_query_based_acc_4, bwd_query_based_std_4, {"color":"blue", "linestyle": "-",  "marker": "x"}),
            
            },
             title="Permuted CIFAR 100 - Query only attention with balanced task buffer - 140",)



lr_index, runs, NUM_TASKS, average_over = "0" , ["0"] , 50000, 10

loss_query_based_4, loss_query_based_std_4  = calculate_curve(base_path, runs, lr_index, "train_loss",  NUM_TASKS, average_over)

plot_graph({ 
            "Forward Accurcy":  (loss_query_based_4, loss_query_based_std_4, {"color":"red", "linestyle": "-",  "marker": "d"}),
            
            },
             title="Permuted CIFAR 100 - Query only attention with balanced task buffer - 140",)










