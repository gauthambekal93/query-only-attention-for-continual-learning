import os
import sys
sys.path.append("C:/Users/gauthambekal93/Research/continual_learning/loss_of_plasticity_and_forgetting/common/codes")
import json
import copy
import argparse
import subprocess
from miscellaneous import *
#import lop
from tqdm import tqdm

import glob
import shutil

def main(arguments):
    cwd = os.getcwd()
    
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment", type=str, default='cfg/a.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)
   
    """These are permutations of parameters for which we need to create run time configuration files """
    param_names, param_combinations = get_configurations(params=params)
    
    """Creates a folder to store models and results"""    
    os.makedirs( os.path.join(project_root, params['model_dir']), exist_ok=True)
    
     
    """Creates a folder to store run time configurations"""    
    run_time_dir = os.path.join(project_root, params["run_time_config_dir"])
    
    os.makedirs(run_time_dir, exist_ok=True)
    
    
    counter =0
    
    for combination_id, param_combination in enumerate( param_combinations):
        
        for run_num in range (params['num_runs']):
            
            run_time_params = copy.deepcopy(params)
            
            for i in range (len(param_combination)):
                
                run_time_params [ param_names[i] ] = param_combination[i]
            
            """Dynamically create result directory to save models and other logs """    
            model_directory = os.path.join(project_root, run_time_params['model_dir'], str(run_num) ,str(combination_id) )
            
            os.makedirs(model_directory , exist_ok=True)
            
            run_time_params['model_dir'] = os.path.join(run_time_params["model_dir"], str(run_num) ,str(combination_id)  )
            
            """Create runtime configuration file """
            run_time_config_path = os.path.join(project_root, params["run_time_config_dir"], str(counter)+'.json')
           
            run_time_params['run_time_config_path'] = os.path.join(run_time_params["run_time_config_dir"], str(counter)+'.json' )
            

            with open(run_time_config_path, 'w+') as f:
                
                json.dump(run_time_params, f, indent=4)
             
            counter += 1  
            

    
if __name__ == '__main__':
    
    project_root = os.path.abspath( os.path.join(os.getcwd(), "..","..", ".."))
    
    config_path = os.path.join(project_root, "configuration_files","models", "maml_permuted_mnist.json")
    
    #config_path = os.path.join(project_root, "configuration_files","models", "bp_permuted_mnist.json")
        
    sys.exit( main ( ['-c', config_path ] ) )
    