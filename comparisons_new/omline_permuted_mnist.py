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

def calculate_curve(model_type, run_numbers, lr_index, key, num_tasks=None, average_over = 0, sub_model_type = None):
    """
    Loads output.pkl from multiple seeds (run_numbers) and returns mean curve over runs.
    key: 'forward_accuracies' | 'backward_accuracies' | 'forward_effective_ranks' | ...
    num_tasks: optional truncate length
    """
    runs = []
    for rn in run_numbers:
        if sub_model_type is None:
            p = os.path.join(project_root, "results", "permuted_mnist", model_type, "fifo_replay", rn, lr_index, "result.pkl")
        else:
            p = os.path.join(project_root, "results", "permuted_mnist", model_type, sub_model_type, rn, lr_index, "result.pkl")
        out = _load_pickle(p)
        
        if model_type=="experience_replay" and rn=="4" and lr_index=="0" or rn=="7" and key!="backward_accuracy":
            arr = np.array([v.item() for k, v in out[key].items()])   [: num_tasks] 
        else:
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


'''

lr_index, runs, NUM_TASKS, average_over = "0" , ["0"] , 12000, 1000

experience_replay, er_forward_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)

# Overall accuracy
plot_graph({"Experience Replay Replay":  (experience_replay, er_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}), },
             title="Permuted MNIST 100 — Performance",)


lr_index, runs, NUM_TASKS, average_over = "0" , ["1"] , 12000, 1000

experience_replay, er_forward_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)

# Overall accuracy
plot_graph({"Experience Replay Replay":  (experience_replay, er_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}), },
             title="Permuted MNIST 100 — Performance",)

lr_index, runs, NUM_TASKS, average_over = "0" , ["2"] , 12000, 1000

experience_replay, er_forward_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)

# Overall accuracy
plot_graph({"Experience Replay Replay":  (experience_replay, er_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}), },
             title="Permuted MNIST 100 — Performance",)


lr_index, runs, NUM_TASKS, average_over = "0" , ["3"] , 12000, 100

experience_replay, er_forward_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)

# Overall accuracy
plot_graph({"Experience Replay Replay":  (experience_replay, er_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}), },
             title="Permuted MNIST 100 — Performance",)


lr_index, runs, NUM_TASKS, average_over = "0" , ["5"] , 12000, 20

experience_replay, er_forward_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)

# Overall accuracy
plot_graph({"Experience Replay Replay":  (experience_replay, er_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}), },
             title="Permuted MNIST 100 — Performance",)




lr_index, runs, NUM_TASKS, average_over = "0" , ["7"] , 12000, 20

experience_replay, er_forward_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)

# Overall accuracy
plot_graph({"Experience Replay Replay":  (experience_replay, er_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}), },
             title="Permuted MNIST 100 — Performance",)


lr_index, runs, NUM_TASKS, average_over = "0" , ["8"] , 12000, 20

experience_replay, er_forward_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)

# Overall accuracy
plot_graph({"Experience Replay Replay":  (experience_replay, er_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}), },
             title="Permuted MNIST 100 — Performance",)


lr_index, runs, NUM_TASKS, average_over = "0" , ["9"] , 12000, 20

experience_replay, er_forward_std  = calculate_curve("experience_replay", runs, lr_index, "last_few_accuracy",  NUM_TASKS, average_over)

# Overall accuracy
plot_graph({"Experience Replay Replay":  (experience_replay, er_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}), },
             title="Permuted MNIST 100 — Performance",)



lr_index, runs, NUM_TASKS, average_over = "0" , ["10"] , 50000, 20

experience_replay, er_forward_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)

# Overall accuracy
plot_graph({"Experience Replay Replay":  (experience_replay, er_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}), },
             title="Permuted MNIST 100 — Performance",)

lr_index, runs, NUM_TASKS, average_over = "0" , ["10"] , 12000, 20

experience_replay, er_forward_std  = calculate_curve("experience_replay", runs, lr_index, "backward_accuracy",  NUM_TASKS, average_over)

# Overall accuracy
plot_graph({"Experience Replay Replay":  (experience_replay, er_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}), },
             title="Permuted MNIST 100 — Performance",)

lr_index, runs, NUM_TASKS, average_over = "0" , ["10"] , 12000, 20

experience_replay, er_forward_std  = calculate_curve("experience_replay", runs, lr_index, "prequential_accuracy",  NUM_TASKS, average_over)

# Overall accuracy
plot_graph({"Experience Replay Replay":  (experience_replay, er_forward_std, {"color":"black", "linestyle": "-",  "marker": "d"}), },
             title="Permuted MNIST 100 — Performance",)

'''


lr_index, runs, NUM_TASKS, average_over = "0" , ["11"] , 50000, 20

fwd_er_acc, fwd_er_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)

prequential_er_acc, prequential_er_std  = calculate_curve("experience_replay", runs, lr_index, "prequential_accuracy",  NUM_TASKS, average_over)
bwd_er_acc, bwd_er_std  = calculate_curve("experience_replay", runs, lr_index, "backward_accuracy",  NUM_TASKS, average_over)

plot_graph({ 
            "Forward Accuracy":  (fwd_er_acc, fwd_er_std, {"color":"red", "linestyle": "-",  "marker": "d"}),
            "Prequential Accuracy":  (prequential_er_acc, prequential_er_std, {"color":"green", "linestyle": "-",  "marker": "s"}),
            "Backward Accuracy":  (bwd_er_acc, bwd_er_std, {"color":"blue", "linestyle": "-",  "marker": "s"}),
            },
             title="Permuted MNIST - Experience replay FIFO - 300000",)



lr_index, runs, NUM_TASKS, average_over = "0" , ["12"] , 50000, 20

fwd_er_acc, fwd_er_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)

prequential_er_acc, prequential_er_std  = calculate_curve("experience_replay", runs, lr_index, "prequential_accuracy",  NUM_TASKS, average_over)

bwd_er_acc, bwd_er_std  = calculate_curve("experience_replay", runs, lr_index, "backward_accuracy",  NUM_TASKS, average_over)

plot_graph({ 
            "Forward Accuracy":  (fwd_er_acc, fwd_er_std, {"color":"red", "linestyle": "-",  "marker": "d"}),
            "Prequential Accuracy":  (prequential_er_acc, prequential_er_std, {"color":"green", "linestyle": "-",  "marker": "s"}),
            "Backward Accuracy":  (bwd_er_acc, bwd_er_std, {"color":"blue", "linestyle": "-",  "marker": "s"}),
            },
             title="Permuted MNIST - Experience replay FIFO - 2000",)





lr_index, runs, NUM_TASKS, average_over = "0" , ["13"] , 50000, 20

fwd_er_acc, fwd_er_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over)

prequential_er_acc, prequential_er_std  = calculate_curve("experience_replay", runs, lr_index, "prequential_accuracy",  NUM_TASKS, average_over)

bwd_er_acc, bwd_er_std  = calculate_curve("experience_replay", runs, lr_index, "backward_accuracy",  NUM_TASKS, average_over)

plot_graph({ 
            "Forward Accuracy":  (fwd_er_acc, fwd_er_std, {"color":"red", "linestyle": "-",  "marker": "d"}),
            "Prequential Accuracy":  (prequential_er_acc, prequential_er_std, {"color":"green", "linestyle": "-",  "marker": "s"}),
            "Backward Accuracy":  (bwd_er_acc, bwd_er_std, {"color":"blue", "linestyle": "-",  "marker": "s"}),
            },
             title="Permuted MNIST - Experience replay FIFO - 20000",)




lr_index, runs, NUM_TASKS, average_over = "0" , ["0"] , 50000, 100

fwd_er_acc, fwd_er_std  = calculate_curve("experience_replay", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over, sub_model_type  = "reservoir_replay")

prequential_er_acc, prequential_er_std  = calculate_curve("experience_replay", runs, lr_index, "prequential_accuracy",  NUM_TASKS, average_over, sub_model_type  = "reservoir_replay")

bwd_er_acc, bwd_er_std  = calculate_curve("experience_replay", runs, lr_index, "backward_accuracy",  NUM_TASKS, average_over, sub_model_type  = "reservoir_replay")


# Overall accuracy
plot_graph({ 
            "Forward Accuracy":  (fwd_er_acc, fwd_er_std, {"color":"red", "linestyle": "-",  "marker": "d"}),
            "Prequential Accuracy":  (prequential_er_acc, prequential_er_std, {"color":"green", "linestyle": "-",  "marker": "s"}),
            "Backward Accuracy":  (bwd_er_acc, bwd_er_std, {"color":"blue", "linestyle": "-",  "marker": "s"}),
            },
             title="Permuted MNIST - Experience replay Reservoir sampling - 2000",)




lr_index, runs, NUM_TASKS, average_over = "0" , ["0"] , 50000, 100

fwd_query_acc, fwd_query_std  = calculate_curve("query_based_cl", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over, sub_model_type  = "reservoir_replay")

prequential_query_acc, prequential_query_std  = calculate_curve("query_based_cl", runs, lr_index, "prequential_accuracy",  NUM_TASKS, average_over, sub_model_type  = "reservoir_replay")

bwd_query_acc, bwd_query_std  = calculate_curve("query_based_cl", runs, lr_index, "backward_accuracy",  NUM_TASKS, average_over, sub_model_type  = "reservoir_replay")


# Overall accuracy
plot_graph({ 
            "Forward Accuracy":  (fwd_query_acc, fwd_query_std, {"color":"red", "linestyle": "-",  "marker": "d"}),
            "Prequential Accuracy":  (prequential_query_acc, prequential_query_std, {"color":"green", "linestyle": "-",  "marker": "s"}),
            "Backward Accuracy":  (bwd_query_acc, bwd_query_std, {"color":"blue", "linestyle": "-",  "marker": "s"}),
            },
             title="Permuted MNIST - Query Based CL Reservoir sampling - 2000",)




lr_index, runs, NUM_TASKS, average_over = "0" , ["0"] , 50000, 100

fwd_query_acc, fwd_query_std  = calculate_curve("query_based_cl", runs, lr_index, "forward_accuracy",  NUM_TASKS, average_over, sub_model_type  = "task_balanced_replay")

prequential_query_acc, prequential_query_std  = calculate_curve("query_based_cl", runs, lr_index, "prequential_accuracy",  NUM_TASKS, average_over, sub_model_type  = "task_balanced_replay")

bwd_query_acc, bwd_query_std  = calculate_curve("query_based_cl", runs, lr_index, "backward_accuracy",  NUM_TASKS, average_over, sub_model_type  = "task_balanced_replay")


# Overall accuracy
plot_graph({ 
            "Forward Accuracy":  (fwd_query_acc, fwd_query_std, {"color":"red", "linestyle": "-",  "marker": "d"}),
            "Prequential Accuracy":  (prequential_query_acc, prequential_query_std, {"color":"green", "linestyle": "-",  "marker": "s"}),
            "Backward Accuracy":  (bwd_query_acc, bwd_query_std, {"color":"blue", "linestyle": "-",  "marker": "s"}),
            },
             title="Permuted MNIST - Query Based CL Task Balanced sampling - 2000",)

# === CAPTION SUGGESTION =======================================================
# “Curves show mean over 3 seeds; markers every 50 tasks. The MAML-style variant
# (ours) is trained for 100 tasks due to compute and is plotted with a dashed black
# line; the vertical line marks its training horizon.”
