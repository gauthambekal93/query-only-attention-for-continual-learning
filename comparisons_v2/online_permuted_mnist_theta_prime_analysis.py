# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:39:50 2026

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


def calculate_curve( base_path, config_id, seed_ids, key, num_tasks=None, average_over = 0):
    """
    Loads output.pkl from multiple seeds (run_numbers) and returns mean curve over runs.
    key: 'forward_accuracies' | 'backward_accuracies' | 'forward_effective_ranks' | ...
    num_tasks: optional truncate length
    """
    runs = []
    
    for seed_id in seed_ids:
        
        p = os.path.join(base_path, config_id, seed_id, "result.pkl" )
        
        out = _load_pickle(p)
        arr = np.array([v for k, v in out[key].items()])   [: num_tasks] 
        arr = ( arr - arr.min() ) /  ( arr.max() - arr.min() )
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

    

def plot_graph_on_axis(ax, series_dict, title, ylabel):

    for lbl, payload in series_dict.items():
        y, std, style = payload

        x = np.arange(len(y))

        ax.plot(x, y, label=lbl, color=style["color"])

        ax.fill_between(x, y - std, y + std,
                        color=style["color"], alpha=0.1)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Steps", fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.legend()


# ==============query-based cl====================


config_id, seed_ids, NUM_TASKS, average_over = "8" , ["0", "1", "2"] , 1990, 100
base_path = os.path.join(project_root, "results", "permuted_mnist", "query_based_cl", "fifo_buffer")

deltas_qcl_acc, deltas_qcl_std  = calculate_curve(base_path, config_id, seed_ids, "deltas",  NUM_TASKS, average_over)
prequential_qcl_acc, prequential_qcl_std  = calculate_curve(base_path, config_id, seed_ids, "prequential_accuracy",  NUM_TASKS, average_over)


# ==============full_attention====================


config_id, seed_ids, NUM_TASKS, average_over = "4" , ["0", "1", "2"] , 1990, 100
base_path = os.path.join(project_root, "results", "permuted_mnist", "full_attention")

deltas_fatt_acc, deltas_fatt_std  = calculate_curve(base_path, config_id, seed_ids, "deltas",  NUM_TASKS, average_over)
prequential_fatt_acc, prequential_fatt_std  = calculate_curve(base_path, config_id, seed_ids, "prequential_accuracy",  NUM_TASKS, average_over)





fig, axes = plt.subplots(1, 2, figsize=(9, 5)) 

# LEFT → Accuracy
plot_graph_on_axis(
    axes[0],
    {
        "Q_CL": (prequential_qcl_acc, prequential_qcl_std, {"color": "black"}),
        "Full_attn": (prequential_fatt_acc, prequential_fatt_std, {"color": "red"})
    },
    "Prequential Accuracy",
    "Accuracy"
)

# RIGHT → Theta Prime
plot_graph_on_axis(
    axes[1],
    {
        "Q_CL": (deltas_qcl_acc, deltas_qcl_std, {"color": "grey"}),
        "Full_attn": (deltas_fatt_acc, deltas_fatt_std, {"color": "#f4a3a3"})
    },
    "Constraint Gap Over Time",
    "Constraint Gap"
)

plt.tight_layout()
plt.show()
