import yaml
import itertools
from pathlib import Path
import copy
import os

def generate_experiments(config_path=None):

    if config_path is None:
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_config = config.get('base_config', {})
    all_experiments = []

    for exp_group in config.get('experiments', []):
        group_name = exp_group['name']
        tag = exp_group.get('tag')

        if 'params' in exp_group:
            param_keys = list(exp_group['params'].keys())
            param_values = list(exp_group['params'].values())
            
            for values_combination in itertools.product(*param_values):
                run_config = copy.deepcopy(base_config)
                run_config.update(exp_group.get('fixed_params', {}))
                
                current_params = dict(zip(param_keys, values_combination))
                run_config.update(current_params)
                
                run_name_parts = [tag] if tag else [group_name]
                for key, val in current_params.items():
                    run_name_parts.append(f"{key.replace('_', '')}{val}")
                
                run_config['run_name'] = "_".join(run_name_parts)
                run_config['tag'] = tag
                all_experiments.append(run_config)

        elif 'runs' in exp_group:
            for run in exp_group['runs']:
                run_config = copy.deepcopy(base_config)
                
                run_params = copy.deepcopy(run.get('params', {}))
                for key, value in run_params.items():
                    if isinstance(value, dict) and key in run_config and isinstance(run_config[key], dict):
                        run_config[key].update(value)
                    else:
                        run_config[key] = value

                run_config['run_name'] = run['name']
                run_config['tag'] = tag
                if 'num_gpus' in run:
                    run_config['num_gpus'] = run['num_gpus']
                all_experiments.append(run_config)

    return all_experiments
