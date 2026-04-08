# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:27:22 2026

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



#def calculate_curve( base_path, run_numbers, lr_index, key, num_tasks=None, average_over = 0):
def calculate_curve( base_path, config_id, seed_ids, key, num_tasks=None, average_over = 0):    
    """
    Loads output.pkl from multiple seeds (run_numbers) and returns mean curve over runs.
    key: 'forward_accuracies' | 'backward_accuracies' | 'forward_effective_ranks' | ...
    num_tasks: optional truncate length
    """
    
    #try:
        
        
    runs = []
    for seed_id in seed_ids:
        p = os.path.join(base_path, config_id, seed_id, "result.pkl" )
        out = _load_pickle(p)
        arr = np.array([v for k, v in out[key].items()])   [ : num_tasks] 
        runs.append(arr)
    #min_len = min(len(runs[0]), len(runs[1]), len(runs[2]))
    #runs[0] = runs[0][:min_len]
    #runs[1] = runs[1][:min_len]
    #runs[2] = runs[2][:min_len]
    
           
    runs = np.stack(runs, axis=1)  # [T, R]
    mean_curve = runs.mean(axis=1)  # [T]
    std  = runs.std(axis = 1)  
    
    
    """Block avg """
    mean_curve = np.array( [np.mean(mean_curve[i: i+ average_over]) for i in range (0, len (mean_curve), average_over )])
    std = np.array([np.mean(std[i: i+ average_over]) for i in range (0, len (std), average_over )])
    "Running avg """
            #mean_curve = np.array( [np.mean(mean_curve[i:i+average_over]) for i in range(0, len(mean_curve), average_over)])
            #std = np.array( [np.mean(std[i:i+average_over]) for i in range(0, len(std), average_over)])
    
    #except:
    #    print("debug")
    #    print("debug")
            
    
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

#def plot_rank_graph(series_dict, title):
#    plot_graph(series_dict, title=title, ylabel="Effective rank")

# === LOAD CURVES ==============================================================

# Common settings




# ===============BP============

config_id, seed_ids, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 1000, 20

base_path = os.path.join(project_root, "results", "slowly_changing_regression", "bp")

train_loss_bp, train_loss_std_bp  = calculate_curve(base_path, config_id, seed_ids, "train_loss",  NUM_TASKS, average_over)
forward_loss_bp, forward_loss_std_bp = calculate_curve(base_path, config_id, seed_ids, "forward_loss",  NUM_TASKS, average_over)
prequential_loss_bp, prequential_loss_std_bp = calculate_curve(base_path, config_id, seed_ids, "prequential_loss",  NUM_TASKS, average_over)
bwd_loss_bp, bwd_loss_std_bp  = calculate_curve(base_path,  config_id, seed_ids, "backward_loss",  NUM_TASKS, average_over)


# ===============CBP============

config_id, seed_ids, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 1000, 20

base_path = os.path.join(project_root, "results", "slowly_changing_regression", "cbp")

train_loss_cbp, train_loss_std_cbp  = calculate_curve(base_path, config_id, seed_ids, "train_loss",  NUM_TASKS, average_over)
forward_loss_cbp, forward_loss_std_cbp = calculate_curve(base_path, config_id, seed_ids, "forward_loss",  NUM_TASKS, average_over)
prequential_loss_cbp, prequential_loss_std_cbp = calculate_curve(base_path, config_id, seed_ids, "prequential_loss",  NUM_TASKS, average_over)
bwd_loss_cbp, bwd_loss_std_cbp  = calculate_curve(base_path,  config_id, seed_ids, "backward_loss",  NUM_TASKS, average_over)


# ===============EWC============

config_id, seed_ids, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 1000, 20

base_path = os.path.join(project_root, "results", "slowly_changing_regression", "ewc")

train_loss_ewc, train_loss_std_ewc  = calculate_curve(base_path, config_id, seed_ids, "train_loss",  NUM_TASKS, average_over)
forward_loss_ewc, forward_loss_std_ewc = calculate_curve(base_path, config_id, seed_ids, "forward_loss",  NUM_TASKS, average_over)
prequential_loss_ewc, prequential_loss_std_ewc = calculate_curve(base_path, config_id, seed_ids, "prequential_loss",  NUM_TASKS, average_over)
bwd_loss_ewc, bwd_loss_std_ewc  = calculate_curve(base_path,  config_id, seed_ids, "backward_loss",  NUM_TASKS, average_over)

# ===============Regenerative Regularisation============

config_id, seed_ids, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 1000, 20

base_path = os.path.join(project_root, "results", "slowly_changing_regression", "regenerative_regularization")

train_loss_regen_reg, train_loss_std_regen_reg  = calculate_curve(base_path, config_id, seed_ids, "train_loss",  NUM_TASKS, average_over)
forward_loss_regen_reg, forward_loss_std_regen_reg = calculate_curve(base_path, config_id, seed_ids, "forward_loss",  NUM_TASKS, average_over)
prequential_loss_regen_reg, prequential_loss_std_regen_reg = calculate_curve(base_path, config_id, seed_ids, "prequential_loss",  NUM_TASKS, average_over)
bwd_loss_regen_reg, bwd_loss_std_regen_reg  = calculate_curve(base_path,  config_id, seed_ids, "backward_loss",  NUM_TASKS, average_over)

# ===============Concat_ReLU============

config_id, seed_ids, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 1000, 20

base_path = os.path.join(project_root, "results", "slowly_changing_regression", "concat_relu")

train_loss_concat_relu, train_loss_std_concat_relu  = calculate_curve(base_path, config_id, seed_ids, "train_loss",  NUM_TASKS, average_over)
forward_loss_concat_relu, forward_loss_std_concat_relu = calculate_curve(base_path, config_id, seed_ids, "forward_loss",  NUM_TASKS, average_over)
prequential_loss_concat_relu, prequential_loss_std_concat_relu = calculate_curve(base_path, config_id, seed_ids, "prequential_loss",  NUM_TASKS, average_over)
bwd_loss_concat_relu, bwd_loss_std_concat_relu  = calculate_curve(base_path,  config_id, seed_ids, "backward_loss",  NUM_TASKS, average_over)

# ===============Experience Replay============

config_id, seed_ids, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 1000, 20

base_path = os.path.join(project_root, "results", "slowly_changing_regression", "experience_replay", "reservoir_replay")

train_loss_exp_rep, train_loss_std_exp_rep  = calculate_curve(base_path, config_id, seed_ids, "train_loss",  NUM_TASKS, average_over)
forward_loss_exp_rep, forward_loss_std_exp_rep = calculate_curve(base_path, config_id, seed_ids, "forward_loss",  NUM_TASKS, average_over)
prequential_loss_exp_rep, prequential_loss_std_exp_rep = calculate_curve(base_path, config_id, seed_ids, "prequential_loss",  NUM_TASKS, average_over)
bwd_loss_exp_rep, bwd_loss_std_exp_rep  = calculate_curve(base_path,  config_id, seed_ids, "backward_loss",  NUM_TASKS, average_over)


# ===============Dark Experience Replay============

config_id, seed_ids, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 1000, 20

base_path = os.path.join(project_root, "results", "slowly_changing_regression", "dark_experience_replay", "reservoir_replay")

train_loss_dark_exp_rep, train_loss_std__dark_exp_rep  = calculate_curve(base_path, config_id, seed_ids, "train_loss",  NUM_TASKS, average_over)
forward_loss_dark_exp_rep, forward_loss_std_dark_exp_rep = calculate_curve(base_path, config_id, seed_ids, "forward_loss",  NUM_TASKS, average_over)
prequential_loss_dark_exp_rep, prequential_loss_std_dark_exp_rep = calculate_curve(base_path, config_id, seed_ids, "prequential_loss",  NUM_TASKS, average_over)
bwd_loss_dark_exp_rep, bwd_loss_std_dark_exp_rep  = calculate_curve(base_path,  config_id, seed_ids, "backward_loss",  NUM_TASKS, average_over)


# ===============Query Based CL============


config_id, seed_ids, NUM_TASKS, average_over = "0" , [ "3"] , 1000, 20

base_path = os.path.join(project_root, "results", "slowly_changing_regression", "query_based_cl","fifo_buffer")

train_loss_query_based_fifo, train_loss_std_query_based_fifo = calculate_curve(base_path, config_id, seed_ids, "train_loss",  NUM_TASKS, average_over)
forward_loss_query_based_fifo, forward_loss_std_query_based_fifo = calculate_curve(base_path, config_id, seed_ids, "forward_loss",  NUM_TASKS, average_over)
prequential_loss_query_based_fifo, prequential_loss_std_query_based_fifo = calculate_curve(base_path, config_id, seed_ids, "prequential_loss",  NUM_TASKS, average_over)
bwd_loss_query_based_fifo, bwd_loss_std_query_based_fifo  = calculate_curve(base_path, config_id, seed_ids, "backward_loss",  NUM_TASKS, average_over)



plot_graph({ 
            
            "BP: Prequential Loss":  (prequential_loss_bp, prequential_loss_std_bp, {"color":"red", "linestyle": "-",  "marker": "s"}),
            "CBP: Prequential Loss":  (prequential_loss_cbp, prequential_loss_std_cbp, {"color":"green", "linestyle": "-",  "marker": "s"}),
            "EWC: Prequential Loss":  (prequential_loss_ewc, prequential_loss_std_ewc, {"color":"purple", "linestyle": "-",  "marker": "s"}),
            "Regen_Reg: Prequential Loss":  (prequential_loss_regen_reg, prequential_loss_std_regen_reg, {"color":"hotpink", "linestyle": "-",  "marker": "s"}),
            "Concat_ReLU: Prequential Loss":  (prequential_loss_concat_relu, prequential_loss_std_concat_relu, {"color":"peachpuff", "linestyle": "-",  "marker": "s"}),
            "Exp_Rep: Prequential Loss":  (prequential_loss_exp_rep, prequential_loss_std_exp_rep, {"color":"blue", "linestyle": "-",  "marker": "s"}),
            "Dark_Exp_Rep: Prequential Loss":  (prequential_loss_dark_exp_rep, prequential_loss_std_dark_exp_rep, {"color":"orange", "linestyle": "-",  "marker": "s"}),
            "Query_CL: Prequential Loss":  (prequential_loss_query_based_fifo, prequential_loss_std_query_based_fifo, {"color":"black", "linestyle": "-",  "marker": "s"}),
          
            },
             title="Slowly Changing Regression - Prequential Loss",
             ylabel = "Loss")

plot_graph({ 
            
            "BP: Forward Loss":  (forward_loss_bp, forward_loss_std_bp, {"color":"red", "linestyle": "-",  "marker": "s"}),
            "CBP: Forward Loss":  (forward_loss_cbp, forward_loss_std_cbp, {"color":"green", "linestyle": "-",  "marker": "s"}),
            "EWC: Forward Loss":  (forward_loss_ewc, forward_loss_std_ewc, {"color":"purple", "linestyle": "-",  "marker": "s"}),
            "Regen_Reg: Prequential Loss":  (forward_loss_regen_reg, forward_loss_std_regen_reg, {"color":"hotpink", "linestyle": "-",  "marker": "s"}),
            "Concat_ReLU: Prequential Loss":  (forward_loss_concat_relu, forward_loss_std_concat_relu, {"color":"peachpuff", "linestyle": "-",  "marker": "s"}),
            "Exp_Rep: Prequential Loss":  (forward_loss_exp_rep, forward_loss_std_exp_rep, {"color":"blue", "linestyle": "-",  "marker": "s"}),
            "Dark_Exp_Rep: Prequential Loss":  (forward_loss_dark_exp_rep, forward_loss_std_dark_exp_rep, {"color":"orange", "linestyle": "-",  "marker": "s"}),
            "Query_CL: Forward Loss":  (forward_loss_query_based_fifo, forward_loss_query_based_fifo, {"color":"black", "linestyle": "-",  "marker": "s"}),

            },
             title="Slowly Changing Regression - Forward Loss",
             ylabel = "Loss")

'''

""" ===============Query Based CL============"""




config_id, seed_ids, NUM_TASKS, average_over = "0" , ["0"] , 1000, 50
>>>>>>> main
base_path = os.path.join(project_root, "results", "slowly_changing_regression", "query_based_cl","fifo_buffer")

train_loss_query_based_fifo, train_loss_std_query_based_fifo = calculate_curve(base_path, config_id, seed_ids, "train_loss",  NUM_TASKS, average_over)
forward_loss_query_based_fifo, forward_loss_std_query_based_fifo = calculate_curve(base_path, config_id, seed_ids, "forward_loss",  NUM_TASKS, average_over)
prequential_loss_query_based_fifo, prequential_loss_std_query_based_fifo = calculate_curve(base_path, config_id, seed_ids, "prequential_loss",  NUM_TASKS, average_over)
bwd_loss_query_based_fifo, bwd_loss_std_query_based_fifo  = calculate_curve(base_path, config_id, seed_ids, "backward_loss",  NUM_TASKS, average_over)



plot_graph({ 
            
            "BP: Prequential Loss":  (prequential_loss_bp, prequential_loss_std_bp, {"color":"blue", "linestyle": "-",  "marker": "s"}),
            "CBP: Prequential Loss":  (prequential_loss_cbp, prequential_loss_std_cbp, {"color":"red", "linestyle": "-",  "marker": "s"}),
            "EWC: Prequential Loss":  (prequential_loss_ewc, prequential_loss_std_ewc, {"color":"yellow", "linestyle": "-",  "marker": "s"}),
            "Query Based FIFO: Prequential Loss":  (prequential_loss_query_based_fifo, prequential_loss_std_query_based_fifo, {"color":"black", "linestyle": "-",  "marker": "s"}),
            },
             title="Slowly Changing Regression - Prequential Loss",
             ylabel = "Loss")

plot_graph({ 
            
            "BP: Forward Loss":  (forward_loss_bp, forward_loss_std_bp, {"color":"blue", "linestyle": "-",  "marker": "s"}),
            "CBP: Forward Loss":  (forward_loss_cbp, forward_loss_std_cbp, {"color":"red", "linestyle": "-",  "marker": "s"}),
            "EWC: Forward Loss":  (forward_loss_ewc, forward_loss_std_ewc, {"color":"yellow", "linestyle": "-",  "marker": "s"}),
            "Query Based FIFO: Forward Loss":  (forward_loss_query_based_fifo, forward_loss_std_query_based_fifo, {"color":"black", "linestyle": "-",  "marker": "s"}),
            },
             title="Slowly Changing Regression - Forward Loss",
             ylabel = "Loss")
'''