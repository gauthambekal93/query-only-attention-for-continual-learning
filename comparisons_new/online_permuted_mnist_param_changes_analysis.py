# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:57:48 2026

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
        runs.append(arr)
    
    max_tasks = min([ len(run) for  run  in runs])
    runs = [ run[:max_tasks] for run in runs ]
    
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


def plot_graph(series_dict, title, ylabel, hline_at=None, vline_at=None):
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
    plt.legend(ncol=3, fontsize=11, frameon=True,  loc="upper center", bbox_to_anchor=(0.65, 0.01) )
    plt.tight_layout()
    plt.show()
    
    
    
    

# ==============Vanilla Backprop ==================== 


config_id, seed_ids, NUM_TASKS, average_over = "parameter_adaptations" , [ "0", "1", "2"] , 9000, 100
base_path = os.path.join(project_root, "results", "permuted_mnist", "bp",)

param_change_bp_acc, param_change_bp_std  = calculate_curve(base_path,  config_id, seed_ids, "param_change",  NUM_TASKS, average_over)



# ==============query-based cl====================


config_id, seed_ids, NUM_TASKS, average_over = "parameter_adaptations" , ["0"] , 9000, 100
base_path = os.path.join(project_root, "results", "permuted_mnist", "query_based_cl", "fifo_buffer")

param_change_qcl_acc, param_change_qcl_std  = calculate_curve(base_path, config_id, seed_ids, "param_change",  NUM_TASKS, average_over)

# ==============full_attention====================

config_id, seed_ids, NUM_TASKS, average_over = "parameter_adaptations" , ["0"] , 9000, 100
base_path = os.path.join(project_root, "results", "permuted_mnist", "full_attention")

param_change_fatt_acc, param_change_fatt_std  = calculate_curve(base_path, config_id, seed_ids, "param_change",  NUM_TASKS, average_over)





plot_graph({ 
            "BP: Param Change":  (param_change_bp_acc, param_change_bp_std, {"color":"skyblue", "linestyle": "-",  "marker": "d"}),
            #"CBP: Forward Accuracy":  (fwd_cbp_acc, fwd_cbp_std, {"color":"yellow", "linestyle": "-",  "marker": "s"}),
            #"EWC: Forward Accuracy":  (fwd_ewc_acc, fwd_ewc_std, {"color":"blue", "linestyle": "-",  "marker": "s"}),
            #"Regern_Reg: Forward Accuracy":  (fwd_regen_reg_acc, fwd_regen_reg_std, {"color":"orange", "linestyle": "-",  "marker": "s"}),
            #"Concat_ReLU: Forward Accuracy":  (fwd_concat_relu_acc, fwd_concat_relu_std, {"color":"purple", "linestyle": "-",  "marker": "s"}),
            #"ER: Forward Accurcy":  (fwd_er_replay, fwd_er_std, {"color":"chocolate", "linestyle": "-",  "marker": "x"}),
            #"Dark_Exp: Forward Accuracy":  (fwd_dark_exp_acc, fwd_dark_exp_std, {"color":"magenta", "linestyle": "-",  "marker": "s"}),
            "Q_CL: Param Change":  (param_change_qcl_acc, param_change_qcl_std, {"color":"black", "linestyle": "-",  "marker": "s"}),
            "Full_attention: Param Change":  (param_change_fatt_acc, param_change_fatt_std, {"color":"red", "linestyle": "-",  "marker": "s"}),

            
            },
             title="Augmented Permuted_MNIST - Param Change Over Tasks ",
             ylabel = "Param Change")




    