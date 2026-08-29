# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 09:02:41 2026

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

exp_path = os.path.join(project_root, "results", "consolidated_for_paper", "data_imagenet.pkl")



PLOT_EVERY = 50


def plot_graph(series_dict, title, ylabel, hline_at=None, vline_at=None):
    """
    series_dict: {label: y | label: (y, {style})}
      style keys: color, marker, linewidth, alpha, plot_every, linestyle
    """
    plt.figure(figsize=(8, 6))
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
            marker=None, #style.get("marker", 'o'),
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
            alpha=0.05
        )
        
        
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    if hline_at is not None:
        plt.axhline(hline_at, linewidth=1, alpha=0.5)
    if vline_at is not None:
        plt.axvline(vline_at, linewidth=1, alpha=0.5)
    plt.xlabel("Steps", fontsize = 17)
    plt.ylabel(ylabel, fontsize = 17)
    plt.title(title, fontsize=20)
    plt.grid(True, linewidth=0.3, alpha=0.5)
    plt.legend(ncol=3, fontsize=17, frameon=True,  loc="upper center", bbox_to_anchor=(0.5, -0.14)) 
    plt.tight_layout()
    plt.show()
    
    
    
    
    
with open(exp_path, "rb") as f:
    obj = pickle.load(f)
    
    
for k, v in obj.items():
    print(k, v.shape, type(v))
    
    
plot_graph({ 
            "BP":  (obj['prequential_bp_acc'], obj['prequential_bp_std'], {"color":"skyblue", "linestyle": "-",  "marker": "d"}),
            "CBP":  (obj['prequential_cbp_acc'], obj['prequential_cbp_std'], {"color":"yellow", "linestyle": "-",  "marker": "s"}),
            "EWC":  (obj['prequential_ewc_acc'], obj['prequential_ewc_std'], {"color":"blue", "linestyle": "-",  "marker": "s"}),
            "Regen":  (obj['prequential_regen_reg_acc'], obj['prequential_regen_reg_std'], {"color":"orange", "linestyle": "-",  "marker": "s"}),
            "Concat":  (obj['prequential_concat_relu_acc'], obj['prequential_concat_relu_std'], {"color":"purple", "linestyle": "-",  "marker": "s"}),
            "ER":  (obj['prequential_er_replay'], obj['prequential_er_std'], {"color":"chocolate", "linestyle": "-",  "marker": "x"}),
            "DarkExp":  (obj['prequential_dark_exp_acc'], obj['prequential_dark_exp_std'], {"color":"magenta", "linestyle": "-",  "marker": "s"}),
            "Q_CL":  (obj['prequential_qcl_acc'], obj['prequential_qcl_std'], {"color":"black", "linestyle": "-",  "marker": "s"}),
            "Full-Attn":  (obj['prequential_fatt_acc'], obj['prequential_fatt_std'], {"color":"red", "linestyle": "-",  "marker": "s"}),


            },
             title="Online Image Net 1000 - Prequential Accuracy",
             ylabel = "Accuracy")    


    
plot_graph({ 
            "BP":  (obj['fwd_bp_acc'], obj['fwd_bp_std'], {"color":"skyblue", "linestyle": "-",  "marker": "d"}),
            "CBP":  (obj['fwd_cbp_acc'], obj['fwd_cbp_std'], {"color":"yellow", "linestyle": "-",  "marker": "s"}),
            "EWC":  (obj['fwd_ewc_acc'], obj['fwd_ewc_std'], {"color":"blue", "linestyle": "-",  "marker": "s"}),
            "Regen":  (obj['fwd_regen_reg_acc'], obj['fwd_regen_reg_std'], {"color":"orange", "linestyle": "-",  "marker": "s"}),
            "Concat":  (obj['fwd_concat_relu_acc'], obj['fwd_concat_relu_std'], {"color":"purple", "linestyle": "-",  "marker": "s"}),
            "ER":  (obj['fwd_er_replay'], obj['fwd_er_std'], {"color":"chocolate", "linestyle": "-",  "marker": "x"}),
            "DarkExp":  (obj['fwd_dark_exp_acc'], obj['fwd_dark_exp_std'], {"color":"magenta", "linestyle": "-",  "marker": "s"}),
            "Q_CL":  (obj['fwd_qcl_acc'], obj['fwd_qcl_std'], {"color":"black", "linestyle": "-",  "marker": "s"}),
            "Full-Attn":  (obj['fwd_fatt_acc'], obj['fwd_fatt_std'], {"color":"red", "linestyle": "-",  "marker": "s"}),


            },
             title="Online Image Net 1000- Forward Accuracy",
             ylabel = "Accuracy") 




print("----Online Image Net 1000 ----")
for k, v in obj.items():
    if 'prequential' in k and ( 'acc' in k or 'er_replay' in k):
        start = k.split('_')[0]
        end = k.split('_')[-1]
        k = k.replace(start, '').replace(end, '').replace('_', '')
        
        print("Model Name: ", k, "Start Accuracy: ", v[0], "End Accuracy: ", v[-1])
 



 
optimal_value = 0
for idx in range(len(obj['prequential_bp_acc'])):
    optimal_value = optimal_value + max( obj['prequential_bp_acc'][idx] , obj['prequential_cbp_acc'][idx], obj['prequential_ewc_acc'][idx],
                         obj['prequential_regen_reg_acc'][idx], obj['prequential_concat_relu_acc'][idx], obj['prequential_er_replay'][idx],
                         obj['prequential_dark_exp_acc'][idx], obj['prequential_qcl_acc'][idx],  obj['prequential_fatt_acc'][idx])
    
    
     

print("----Regret ImageNet 1000 ----")
for k, v in obj.items():
    if 'prequential' in k and ( 'acc' in k or 'er_replay' in k):
        start = k.split('_')[0]
        end = k.split('_')[-1]
        k = k.replace(start, '').replace(end, '').replace('_', '')
        
        print("Model Name: ", k,  "Regret: ",  optimal_value - v.sum() )
