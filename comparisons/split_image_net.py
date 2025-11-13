# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 11:03:55 2025

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
#average_over = 50
# Number of tasks to truncate to for the long-running methods
NUM_TASKS_LONG = 9500  # will be clipped to available length per series

# ==============================================================================
def _load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def calculate_curve(model_type, run_numbers, lr_index, key, num_tasks=None, average_over = 0):
    """
    Loads output.pkl from multiple seeds (run_numbers) and returns mean curve over runs.
    key: 'forward_accuracies' | 'backward_accuracies' | 'forward_effective_ranks' | ...
    num_tasks: optional truncate length
    """
    runs = []
    for rn in run_numbers:
        p = os.path.join(project_root, "results", "split_image_net", model_type, rn, lr_index, "output.pkl")
        out = _load_pickle(p)
        arr = np.asarray(out[key], dtype=np.float32)[: num_tasks]
        
        if "maml" in model_type.lower():
            starting_point = np.where(arr==0)[0][-1] + 1
            arr = arr[starting_point:]
            
        runs.append(arr)
        
    runs = np.stack(runs, axis=1)  # [T, R]
    mean_curve = runs.mean(axis=1)  # [T]
    std  = runs.std(axis = 1)  
    
    if 'bp' in model_type:
        print("stop")
        print("stop")
    
    if 'query_based_cl' in model_type:
        print("stop")
        print("stop")
        
    if 'transformer' in model_type:
        print("stop")
        print("stop")

    mean_curve = np.array( [np.mean(mean_curve[i: i+ average_over]) for i in range (0, len (mean_curve), average_over )])
            
    std = np.array([np.mean(std[i: i+ average_over]) for i in range (0, len (std), average_over )])
        
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
        if 'query' in lbl:
            line_width = 2
        else:
            line_width = 1.3
        if 'MAML' in lbl:
            print("stop")
            print("stop")
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

# --- BP ---
lr_index, runs, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 9000, 90 
 
bp_forward_accuracies, bp_forward_std  = calculate_curve("bp", runs, lr_index, "forward_accuracies",  NUM_TASKS, average_over )
bp_backward_accuracies, bp_backward_std = calculate_curve("bp", runs, lr_index, "backward_accuracies", NUM_TASKS, average_over)


# --- CBP ---
lr_index, runs, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 9000, 90 

cbp_forward_accuracies, cbp_forward_std  = calculate_curve("cbp", runs, lr_index, "forward_accuracies",  NUM_TASKS, average_over)
cbp_backward_accuracies, cbp_backward_std = calculate_curve("cbp", runs, lr_index, "backward_accuracies", NUM_TASKS, average_over)

# --- Transformer ---
lr_index, runs, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 9000, 90

transformer_forward_accuracies, transformer_forward_std  = calculate_curve("transformer", runs, lr_index, "forward_accuracies",  NUM_TASKS, average_over)
transformer_backward_accuracies, transformer_backward_std = calculate_curve("transformer", runs, lr_index, "backward_accuracies", NUM_TASKS, average_over)

# --- Query-based CL V1 ---
lr_index, runs, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 9000, 90 

q1_forward_accuracies, q1_forward_std  = calculate_curve("query_based_cl", runs, lr_index, "forward_accuracies",  NUM_TASKS, average_over)
q1_backward_accuracies, q1_backward_std = calculate_curve("query_based_cl", runs, lr_index, "backward_accuracies", NUM_TASKS, average_over)


# ---MAML ---
lr_index, runs, NUM_TASKS, average_over = "0" , ["0", "1", "2"] , 500, 5

maml_forward_accuracies, maml_forward_std = calculate_curve("maml", runs, lr_index, "forward_accuracies",  NUM_TASKS, average_over)
maml_backward_accuracies, maml_backward_std = calculate_curve("maml", runs, lr_index, "backward_accuracies", NUM_TASKS, average_over)


# ---Elastic-weight-compute ---
lr_index, runs, NUM_TASKS, average_over = "0" , ["0","1","2"] , 9000, 90

ewc_forward_accuracies, ewc_forward_std = calculate_curve("elastic_weight_compute", runs, lr_index, "forward_accuracies",  NUM_TASKS, average_over)
ewc_backward_accuracies, ewc_backward_std = calculate_curve("elastic_weight_compute", runs, lr_index, "backward_accuracies", NUM_TASKS, average_over)
# === PLOTS ====================================================================

# Forward accuracy
plot_graph(
    {
        "BP":                         ( bp_forward_accuracies, bp_forward_std, {"color":"red",  "linestyle": "-",  "marker": "o" }, ),
        "CBP":                        (cbp_forward_accuracies, cbp_forward_std, {"color":"blue", "linestyle": "-", "marker": "s"} ),
        "Transformer":                (transformer_forward_accuracies, transformer_forward_std, {"color":"darkgreen",  "linestyle": "-", "marker": "^"}),
        "Query-Only Attention":             (q1_forward_accuracies, q1_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}),
        "MAML":                       (maml_forward_accuracies, maml_forward_std, {"color":"darkorange",  "linestyle": "-",   "marker": "v"}),
        "EWC":                       (ewc_forward_accuracies, ewc_forward_std, {"color":"darkmagenta",  "linestyle": "-",   "marker": "v"}),
  
    },
    title="SPLIT IMAGE NET — Forward Performance",
    
)

# Backward accuracy
plot_graph(
    {
        "BP":                         (bp_backward_accuracies, bp_backward_std, {"color":"red",  "linestyle": "-",  "marker": "o"}),
        "CBP":                        (cbp_backward_accuracies, cbp_backward_std, {"color":"blue", "linestyle": "-", "marker": "s"}),
        "Transformer":                (transformer_backward_accuracies, transformer_backward_std, {"color":"darkgreen",  "linestyle": "-", "marker": "^"}),
        "Query-Only Attention":             (q1_backward_accuracies, q1_backward_std,         {"color":"black", "linestyle": "-",  "marker": "d"}),
        "MAML":                       (maml_backward_accuracies,  maml_backward_std ,   {"color":"darkorange",  "linestyle": "-",   "marker": "v"}),
        "EWC":                       (ewc_backward_accuracies,  ewc_backward_std ,   {"color":"darkmagenta",  "linestyle": "-",   "marker": "v"}),
    },
    title="SPLIT IMAGE NET — Backward Performance",
   
)



# === CAPTION SUGGESTION =======================================================
# “Curves show mean over 3 seeds; markers every 50 tasks. The MAML-style variant
# (ours) is trained for 100 tasks due to compute and is plotted with a dashed black
# line; the vertical line marks its training horizon.”
