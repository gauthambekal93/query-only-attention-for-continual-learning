# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 21:24:48 2025

@author: gauthambekal93
"""

import os
import sys
sys.path.append("C:/Users/gauthambekal93/Research/continual_learning/loss_of_plasticity_and_forgetting/common/codes")
import json
import copy
import argparse
import subprocess
from tqdm import tqdm
from miscellaneous import get_configurations
import shutil


def main(arguments):
    
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment", type=str, default='cfg/a.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    #list_params, hyper_param_settings = get_configurations(params=params)
    
    """Creates a directory to save the dataset """    
    os.makedirs( os.path.join(project_root, params['data_dir']), exist_ok=True)
    
    """Creates a directory to save the configuration files """
    run_time_dir = os.path.join(project_root, params["run_time_config_dir"])
    
    os.makedirs(run_time_dir, exist_ok=True)
    
    for idx in tqdm(range(params['num_runs'])):
        
        run_time_file_path = run_time_dir+'/'+str(idx)+'.json'
        new_params = copy.deepcopy(params)
        new_params['data_dir'] = new_params['data_dir'] + '/'+str(idx)

        with open(run_time_file_path, 'w+') as f:
            json.dump(new_params, f)

if __name__ == '__main__':
    #sys.exit(main(sys.argv[1:]))
    
    project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    
    config_path = os.path.join( project_root, "configuration_files","data","split_image_net", "config.json")
    
    sys.exit( main ( ['-c', config_path ] ) ) #this line create 100 configuration files, for 100 runs inside the env_temp_cfg folder.
    