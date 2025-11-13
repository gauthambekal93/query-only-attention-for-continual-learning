# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 16:58:32 2025

@author: gauthambekal93
"""


# plot_permuted_mnist_all.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# === CONFIG ===================================================================
# Set this to your repo root that contains results/permuted_mnist/<model>/<run>/<lr>/output.pkl
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Plot markers every N points (line still connects all points)
PLOT_EVERY = 50
average_over = 10 #50
# Number of tasks to truncate to for the long-running methods
NUM_TASKS_LONG = 800  # will be clipped to available length per series

# ==============================================================================
def _load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize(runs):
    
    for col_idx in range( runs.shape[1] ):
        run = runs[:, col_idx ]
        min_val = run.min()
        max_val = run.max()
        runs[:, col_idx] = ( run - min_val) / (max_val -  min_val)
        
    return runs

def calculate_curve(model_type, run_numbers, lr_index, key, num_tasks=None):
    """
    Loads output.pkl from multiple seeds (run_numbers) and returns mean curve over runs.
    key: 'forward task mse' | 'backward task mse' | 'forward effective rank' | ...
    num_tasks: optional truncate length
    """
    runs = []
    for rn in run_numbers:
        p = os.path.join(project_root, "results", "slowly_changing_regression", model_type, rn, lr_index, "output.pkl")
        out = _load_pickle(p)
       
        arr = np.asarray(out[key], dtype=np.float32)[: num_tasks]
        
        if "maml" in model_type.lower():
            starting_point = np.where(arr==0)[0][-1] + 1
            arr = arr[starting_point:]
            
        runs.append(arr)
    runs = np.stack(runs, axis=1)  
    
    if "rank" in key.lower():  #only normalize the ranks 
        normalize(runs)
        
    mean_curve = runs.mean(axis=1)
    std  = runs.std(axis = 1)  
    
    '''
    if num_tasks is not None:
        mean_curve = mean_curve[:num_tasks]
        std = std[:num_tasks]
    '''
    
    #if model_type  != "maml":
        
    mean_curve = np.array( [np.mean(mean_curve[i: i+ average_over]) for i in range (0, len (mean_curve), average_over )])
            
    std = np.array([np.mean(std[i: i+ average_over]) for i in range (0, len (std), average_over )])    
   
    return mean_curve, std






def plot_graph(series_dict, title, ylabel="MSE", hline_at=None, vline_at=None):
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
        if 'query' in lbl:
            line_width = 2
        else:
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
    plt.xlabel("Steps", fontsize = 12)
    plt.ylabel(ylabel, fontsize = 12)
    plt.title(title, fontsize=13)
    plt.grid(True, linewidth=0.3, alpha=0.5)
    if 'forward performance' in title.lower():
        plt.legend(ncol=3, fontsize=11, frameon=True,  loc="lower center", bbox_to_anchor=(0.60, 0.80), handletextpad=0.30 )
    else:
        plt.legend(ncol=5, fontsize=11, frameon=True,  loc="lower center", bbox_to_anchor=(0.50, 0.87), handletextpad=0.30 )
        
    plt.tight_layout()
    plt.show()

def plot_rank_graph(series_dict, title):
    plot_graph(series_dict, title=title, ylabel="Effective rank")

# === LOAD CURVES ==============================================================

# Common settings
lr_index = "2"
long_runs = ["0", "1", "2"]  # 3 seeds
short_runs = ["0", "1", "2"] # for MAML (also 3 seeds, but only 100 tasks exist)

# --- BP ---
bp_forward_mse, bp_forward_std  = calculate_curve("bp", long_runs, lr_index, "forward task mse",  NUM_TASKS_LONG)
bp_backward_mse, bp_backward_std = calculate_curve("bp", long_runs, lr_index, "backward task mse", NUM_TASKS_LONG)
bp_forward_ranks, bp_forward_ranks_std  = calculate_curve("bp", long_runs, lr_index, "forward effective rank", NUM_TASKS_LONG)

# --- CBP ---
cbp_forward_mse, cbp_forward_std  = calculate_curve("cbp", long_runs, lr_index, "forward task mse",  NUM_TASKS_LONG)
cbp_backward_mse, cbp_backward_std = calculate_curve("cbp", long_runs, lr_index, "backward task mse", NUM_TASKS_LONG)
cbp_forward_ranks, cbp_forward_ranks_std       = calculate_curve("cbp", long_runs, lr_index, "forward effective rank", NUM_TASKS_LONG)

# --- Transformer ---
transformer_forward_mse, transformer_forward_std  = calculate_curve("transformer", long_runs, lr_index, "forward task mse",  NUM_TASKS_LONG)
transformer_backward_mse, transformer_backward_std = calculate_curve("transformer", long_runs, lr_index, "backward task mse", NUM_TASKS_LONG)
transformer_forward_ranks, transformer_forward_ranks_std       = calculate_curve("transformer", long_runs, lr_index, "forward effective rank", NUM_TASKS_LONG)

# --- Query-based CL V1 ---
q1_forward_mse, q1_forward_std  = calculate_curve("query_based_cl", long_runs, lr_index, "forward task mse",  NUM_TASKS_LONG)
q1_backward_mse, q1_backward_std = calculate_curve("query_based_cl", long_runs, lr_index, "backward task mse", NUM_TASKS_LONG)
q1_forward_ranks, q1_forward_ranks_std       = calculate_curve("query_based_cl", long_runs, lr_index, "forward effective rank", NUM_TASKS_LONG)


# --- MAML-style (ours) — ONLY 100 tasks available ---
#MAML_TASKS = 100
maml_forward_mse, maml_forward_std  = calculate_curve("maml", short_runs, lr_index, "forward task mse",  NUM_TASKS_LONG)
maml_backward_mse, maml_backward_std = calculate_curve("maml", short_runs, lr_index, "backward task mse", NUM_TASKS_LONG)
maml_forward_ranks, maml_forward_ranks_std       = calculate_curve("maml", short_runs, lr_index, "forward effective rank", NUM_TASKS_LONG)

# === PLOTS ====================================================================

# Forward accuracy
plot_graph(
    {
        "BP":                         ( bp_forward_mse, bp_forward_std, {"color":"red"}, ),
        "CBP":                        (cbp_forward_mse, cbp_forward_std, {"color":"blue"} ),
        "Transformer":                (transformer_forward_mse, transformer_forward_std, {"color":"green"}),
        "Query-Only Attention":             (q1_forward_mse, q1_forward_std, {"color":"black"}),
        "MAML": (maml_forward_mse, maml_forward_std,      {"color":"orange"})
    },
    title="SLOWLY CHANGING REGRESSION — Forward Performance",
    vline_at=None #len(maml_forward_mse)-1
)

# Backward accuracy
plot_graph(
    {
        "BP":                         (bp_backward_mse, bp_backward_std, {"color":"red"}),
        "CBP":                        (cbp_backward_mse, cbp_backward_std, {"color":"blue"}),
        "Transformer":                (transformer_backward_mse, transformer_backward_std, {"color":"green"}),
        "Query-Only Attention":             (q1_backward_mse, q1_backward_std,         {"color":"black"}),
        "MAML": (maml_backward_mse, maml_backward_std,   {"color":"orange"})
    },
    title="SLOWLY CHANGING REGRESSION — Backward Performance",
    vline_at=None #len(maml_backward_mse)-1
)

# Forward effective rank
plot_rank_graph(
    {
        "BP":                         (bp_forward_ranks,  bp_forward_ranks_std,  {"color":"red"}),
        "CBP":                        (cbp_forward_ranks, cbp_forward_ranks_std,  {"color":"blue"}),
        "Transformer":                (transformer_forward_ranks, transformer_forward_ranks_std, {"color":"green"}),
        "Query-Only Attention":             (q1_forward_ranks,  q1_forward_ranks_std,         {"color":"black"}),
        "MAML": (maml_forward_ranks, maml_forward_ranks_std,   {"color":"orange"})
    },
    title="SLOWLY CHANGING REGRESSION  — Effective Rank"
)

# === CAPTION SUGGESTION =======================================================
# “Curves show mean over 3 seeds; markers every 50 tasks. The MAML-style variant
# (ours) is trained for 100 tasks due to compute and is plotted with a dashed black
# line; the vertical line marks its training horizon.”
