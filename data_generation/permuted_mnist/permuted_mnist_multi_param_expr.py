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


'''
def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment",
                        type=str, default='cfg/a.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    list_params, hyper_param_settings = get_configurations(params=params)

    # make a directory for temp cfg files
    #bash_command = "mkdir -p temp_cfg/"
    #subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
   
    #os.makedirs("temp_cfg", exist_ok=True)
    
    #bash_command = "rm -r --force " + params['data_dir']
    #subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    
    if os.path.exists(params['data_dir']):
        shutil.rmtree(params['data_dir'])
    
    #bash_command = "mkdir " + params['data_dir']
    #subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    
    os.makedirs(params['data_dir'], exist_ok=True)
    
    """
        Set and write all the parameters for the individual config files
    """
    for setting_index, param_setting in enumerate(hyper_param_settings):
        new_params = copy.deepcopy(params)
        for idx, param in enumerate(list_params):
            new_params[param] = param_setting[idx]
        new_params['index'] = setting_index
        new_params['data_dir'] = params['data_dir'] + str(setting_index) + '/'

        """
            Make the data directory
        """
        #bash_command = "mkdir -p " + new_params['data_dir']
        #subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        
        os.makedirs(new_params['data_dir'], exist_ok=True)
    
        for idx in tqdm(range(params['num_runs'])):
            new_params['data_file'] = new_params['data_dir'] + str(idx)

            """
                write data in config files
            """
            if 'cbp' in path:
                os.makedirs("temp_cfg/cbp/", exist_ok=True)
                new_cfg_file = 'temp_cfg/cbp/'+str(setting_index*params['num_runs']+idx)+'.json'
            else:    
                os.makedirs("temp_cfg/bp/", exist_ok=True)
                new_cfg_file = 'temp_cfg/bp/'+str(setting_index*params['num_runs']+idx)+'.json'
            
            f = open(new_cfg_file, 'w+')
            
            #try:    
            #    f = open(new_cfg_file, 'w+')
            #except: 
            #    f = open(new_cfg_file, 'w+')
            
            with open(new_cfg_file, 'w+') as f:
                json.dump(new_params, f, sort_keys=False, indent=4)
'''

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
    
    config_path = os.path.join( project_root, "configuration_files","data","permuted_mnist", "std_net.json")
    
    sys.exit( main ( ['-c', config_path ] ) ) #this line create 100 configuration files, for 100 runs inside the env_temp_cfg folder.
    