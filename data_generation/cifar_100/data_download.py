# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 18:17:35 2025

@author: gauthambekal93
"""




import os
from pathlib import Path
#import sys
from torchvision import datasets

import json
#BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

BASE_DIR = Path(__file__).resolve().parent.parent.parent


#sys.path.append(str(BASE_DIR / "common" / "codes"))
#sys.path.append ( str(BASE_DIR / "algorithms" / "query_based_cl"/ "Code") )




def main(data_config_path):
   
   with open(data_config_path, 'r') as f:
       data_params = json.load(f)
       
   data_path = os.path.join(BASE_DIR, data_params["data_dir"])
   
   """This line will download the cifar dataset at a given location without loading it to a variable"""
   datasets.CIFAR100(root = data_path, download=True)

if __name__ == '__main__':   
    
    data_config_path = os.path.join(BASE_DIR, "configuration_files","cifar_100", "data", "0.json")

    main(data_config_path)
    
    
#data_path = os.path.join(BASE_DIR, "Research","query-only-attention-for-continual-learning", "data_generation", "cifar_100", "data")





