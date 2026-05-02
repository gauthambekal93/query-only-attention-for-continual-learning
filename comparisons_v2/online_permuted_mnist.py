# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:32:08 2026

@author: gauthambekal93
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import pickle

project_root = Path(__file__).resolve().parent.parent  # go up two levels, adjust as needed

exp_path = os.path.join(project_root, "results", "consolidated_for_paper", "permuted_mnist.pkl")



PLOT_EVERY = 50


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
    
    
    
    
    
with open(exp_path, "rb") as f:
    obj = pickle.load(f)
    
    
for k, v in obj.items():
    print(k, v.shape, type(v))
    
    
plot_graph({ 
            "BP: Prequential Accurcy":  (obj['prequential_bp_acc'], obj['prequential_bp_std'], {"color":"skyblue", "linestyle": "-",  "marker": "d"}),
            "CBP: Prequential Accuracy":  (obj['prequential_cbp_acc'], obj['prequential_cbp_std'], {"color":"yellow", "linestyle": "-",  "marker": "s"}),
            "EWC: Prequential Accuracy":  (obj['prequential_ewc_acc'], obj['prequential_ewc_std'], {"color":"blue", "linestyle": "-",  "marker": "s"}),
            "Regern_Reg: Prequential Accuracy":  (obj['prequential_regen_reg_acc'], obj['prequential_regen_reg_std'], {"color":"orange", "linestyle": "-",  "marker": "s"}),
            "Concat_ReLU: Prequential Accuracy":  (obj['prequential_concat_relu_acc'], obj['prequential_concat_relu_std'], {"color":"purple", "linestyle": "-",  "marker": "s"}),
            "ER: Prequential Accurcy":  (obj['prequential_er_replay'], obj['prequential_er_std'], {"color":"chocolate", "linestyle": "-",  "marker": "x"}),
            "Dark_Exp: Prequential Accuracy":  (obj['prequential_dark_exp_acc'], obj['prequential_dark_exp_std'], {"color":"magenta", "linestyle": "-",  "marker": "s"}),
            "Q_CL: Prequential Accuracy":  (obj['prequential_qcl_acc'], obj['prequential_qcl_std'], {"color":"black", "linestyle": "-",  "marker": "s"}),
            "Full_attention: Prequential Accuracy":  (obj['prequential_fatt_acc'], obj['prequential_fatt_std'], {"color":"red", "linestyle": "-",  "marker": "s"}),


            },
             title="Augmented Permuted_MNIST - Prequential Accuracy",
             ylabel = "Accuracy")    


