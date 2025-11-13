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
average_over = 100
# Number of tasks to truncate to for the long-running methods
NUM_TASKS_LONG = 7500  # will be clipped to available length per series

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
    key: 'forward_accuracies' | 'backward_accuracies' | 'forward_effective_ranks' | ...
    num_tasks: optional truncate length
    """
    runs = []
    for rn in run_numbers:
        p = os.path.join(project_root, "results", "permuted_mnist", model_type, rn, lr_index, "output.pkl")
        out = _load_pickle(p)
        arr = np.asarray(out[key], dtype=np.float32)[: num_tasks]
        runs.append(arr)
    runs = np.stack(runs, axis=1)  # [T, R]
    
    if "rank" in key.lower():  #only normalize the ranks 
        normalize(runs)
        
    mean_curve = runs.mean(axis=1)  # [T]
    std  = runs.std(axis = 1)  
    '''
    if num_tasks is not None:
        mean_curve = mean_curve[:num_tasks]
        std = std[:num_tasks]
    '''
    if model_type  != "maml":
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
            line_width = 1.5
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
    plt.xlabel("Steps", fontsize = 13)
    plt.ylabel(ylabel, fontsize = 13)
    plt.title(title, fontsize=14)
    plt.grid(True, linewidth=0.3, alpha=0.5)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    # insert a newline only for the long ones (keeps full name)
    '''
    labels = [lbl.replace('Query-Only Attention V2', 'Query-Only Attention V2\n')
               .replace('Query-Only Attention V1', 'Query-Only Attention V1\n')
          for lbl in labels]
    
    plt.legend(handles, labels, ncol=3, fontsize=11, frameon=True,  loc="lower center", bbox_to_anchor=(0.65, 0.01) )
    '''
    
    plt.legend(ncol=3, fontsize=11, frameon=True,  loc="lower center", bbox_to_anchor=(0.55, 0.01) )
    
    plt.tight_layout()
    plt.show()

def plot_rank_graph(series_dict, title):
    plot_graph(series_dict, title=title, ylabel="Effective rank")

# === LOAD CURVES ==============================================================

# Common settings
lr_index = "2"
long_runs = ["0", "1", "2"]  # 3 seeds


# --- BP --- #9500 tasks
bp_forward_accuracies, bp_forward_std  = calculate_curve("bp", long_runs, lr_index, "forward_accuracies",  NUM_TASKS_LONG)
bp_backward_accuracies, bp_backward_std = calculate_curve("bp", long_runs, lr_index, "backward_accuracies", NUM_TASKS_LONG)
bp_forward_ranks, bp_forward_ranks_std  = calculate_curve("bp", long_runs, lr_index, "forward_effective_ranks", NUM_TASKS_LONG)
 
# --- CBP --- #9500 tasks
cbp_forward_accuracies, cbp_forward_std  = calculate_curve("cbp", long_runs, lr_index, "forward_accuracies",  NUM_TASKS_LONG)
cbp_backward_accuracies, cbp_backward_std = calculate_curve("cbp", long_runs, lr_index, "backward_accuracies", NUM_TASKS_LONG)
cbp_forward_ranks, cbp_forward_ranks_std       = calculate_curve("cbp", long_runs, lr_index, "forward_effective_ranks", NUM_TASKS_LONG)

# --- Transformer ---  #8950 tasks
transformer_forward_accuracies, transformer_forward_std  = calculate_curve("transformer", long_runs, lr_index, "forward_accuracies",  NUM_TASKS_LONG)
transformer_backward_accuracies, transformer_backward_std = calculate_curve("transformer", long_runs, lr_index, "backward_accuracies", NUM_TASKS_LONG)
transformer_forward_ranks, transformer_forward_std       = calculate_curve("transformer", long_runs, lr_index, "forward_effective_ranks", NUM_TASKS_LONG)

# --- Query-based CL V1 --- #8950 tasks
q1_forward_accuracies, q1_forward_std  = calculate_curve("query_based_cl", long_runs, lr_index, "forward_accuracies",  NUM_TASKS_LONG)
q1_backward_accuracies, q1_backward_std = calculate_curve("query_based_cl", long_runs, lr_index, "backward_accuracies", NUM_TASKS_LONG)
q1_forward_ranks, q1_forward_ranks_std       = calculate_curve("query_based_cl", long_runs, lr_index, "forward_effective_ranks", NUM_TASKS_LONG)

# --- Query-based CL V2 (different run numbers) ---

q2_runs = ["10", "11", "12"]
q2_forward_accuracies, q2_forward_std  = calculate_curve("query_based_cl", q2_runs, lr_index, "forward_accuracies",  NUM_TASKS_LONG)
q2_backward_accuracies, q2_backward_std = calculate_curve("query_based_cl", q2_runs, lr_index, "backward_accuracies", NUM_TASKS_LONG)
q2_forward_ranks, q2_forward_ranks_std       = calculate_curve("query_based_cl", q2_runs, lr_index, "forward_effective_ranks", NUM_TASKS_LONG)

# --- MAML-style (ours) — ONLY 100 tasks available ---
MAML_TASKS = 75
short_runs = ["3", "4"] 
maml_forward_accuracies, maml_forward_std  = calculate_curve("maml", short_runs, lr_index, "forward_accuracies",  MAML_TASKS)
maml_backward_accuracies, maml_backward_std = calculate_curve("maml", short_runs, lr_index, "backward_accuracies", MAML_TASKS)
maml_forward_ranks, maml_forward_ranks_std       = calculate_curve("maml", short_runs, lr_index, "forward_effective_ranks", MAML_TASKS)

# === PLOTS ====================================================================

# Forward accuracy
plot_graph(
    {
        "BP":                         ( bp_forward_accuracies, bp_forward_std, {"color":"red",  "linestyle": "-",  "marker": "o" }, ),
        "CBP":                        (cbp_forward_accuracies, cbp_forward_std, {"color":"blue", "linestyle": "-", "marker": "s"} ),
        "Transformer":                (transformer_forward_accuracies, transformer_forward_std, {"color":"darkgreen",  "linestyle": "-", "marker": "^"}),
        "Query-Only Attention V1":    (q1_forward_accuracies, q1_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}),
        "Query-Only Attention V2":    (q2_forward_accuracies, q2_forward_std, {"color":"brown",  "linestyle": "-",   "marker": "v"}),
        "MAML": (maml_forward_accuracies, maml_forward_std,      {"color":"orange","linestyle":"-", "marker": "x"})
    },
    title="PERMUTED MNIST — Forward Performance",
    #vline_at=len(maml_forward_accuracies)-1
)

# Backward accuracy
plot_graph(
    {
        "BP":                         (bp_backward_accuracies, bp_backward_std, {"color":"red",  "linestyle": "-",  "marker": "o"}),
        "CBP":                        (cbp_backward_accuracies, cbp_backward_std, {"color":"blue", "linestyle": "-", "marker": "s"}),
        "Transformer":                (transformer_backward_accuracies, transformer_backward_std, {"color":"darkgreen",  "linestyle": "-", "marker": "^"}),
        "Query-Only Attention V1":    (q1_backward_accuracies, q1_backward_std,         {"color":"black", "linestyle": "-",  "marker": "d"}),
        "Query-Only Attention V2":    (q2_backward_accuracies,  q2_backward_std ,       {"color":"brown",  "linestyle": "-",   "marker": "v"}),
        "MAML": (maml_backward_accuracies, maml_backward_std,   {"color":"orange","linestyle":"-","marker": "x" })
    },
    title="PERMUTED MNIST — Backward Performance",
    #vline_at=len(maml_backward_accuracies)-1
)

# Forward effective rank
plot_rank_graph(
    {
        "BP":                         (bp_forward_ranks,  bp_forward_ranks_std,  {"color":"red",  "linestyle": "-",  "marker": "o"}),
        "CBP":                        (cbp_forward_ranks, cbp_forward_std,  {"color":"blue","linestyle": "-", "marker": "s"}),
        "Transformer":                (transformer_forward_ranks, transformer_forward_std, {"color":"darkgreen",  "linestyle": "-", "marker": "^"}),
        "Query-Only Attention V1":    (q1_forward_ranks,  q1_forward_std,         {"color":"black", "linestyle": "-",  "marker": "d"}),
        "Query-Only Attention V2":     (q2_forward_ranks,  q2_backward_std,        {"color":"brown",  "linestyle": "-",   "marker": "v"}),
        "MAML": (maml_forward_ranks, maml_forward_ranks_std,   {"color":"orange","linestyle":"-",  "linestyle": "-",  "marker": "x"})
    },
    title="PERMUTED MNIST — Effective Rank"
)

# === CAPTION SUGGESTION =======================================================
# “Curves show mean over 3 seeds; markers every 50 tasks. The MAML-style variant
# (ours) is trained for 100 tasks due to compute and is plotted with a dashed black
# line; the vertical line marks its training horizon.”
