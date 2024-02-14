import json
import subprocess
import os
import argparse
from datetime import datetime
from itertools import product

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments based on a json configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file.")
    parser.add_argument("--output_dir", type=str, default="out", help="Directory to place the set of output checkpoints.")
    parser.add_argument("--prefix", type=str, default='', help="Optional prefix for tensorboard_run_name and out_dir.")
    parser.add_argument("--value_only", action="store_true", help="Include only the values of the configuration parameters in the names.")
    return parser.parse_args()

def check_conditions(conditions, combo_dict):
    return all(combo_dict.get(cond[0]) == cond[1] for cond in conditions)

def expand_range(value):
    if isinstance(value, dict) and 'range' in value:
        range_def = value['range']
        start, end = range_def['start'], range_def['end']
        step = range_def.get('step', 1 if isinstance(start, int) else 0.1)
        return list(range(start, end + 1, step)) if isinstance(start, int) else [start + i * step for i in range(int((end - start) / step) + 1)]
    return value

def generate_combinations(config, current_combo={}, parent_conditions=[]):
    # Initialize base_params with expanded ranges or direct values
    base_params = {}
    for k, v in config.items():
        if isinstance(v, list):
            base_params[k] = v
        elif isinstance(v, dict):
            if 'range' in v:
                # Directly expand ranges for parameters like 'seed'
                base_params[k] = expand_range(v)
            else:
                # Handle conditional parameters separately
                continue
        else:
            base_params[k] = [v]

    conditional_params = {k: v for k, v in config.items() if isinstance(v, dict) and 'conditions' in v}

    base_combinations = list(product(*[[(k, v) for v in values] for k, values in base_params.items()]))

    for base_combination in base_combinations:
        combo_dict = dict(base_combination)

        # Check and apply conditional parameters
        for cond_param, values in conditional_params.items():
            if check_conditions(values['conditions'], combo_dict):
                # This assumes 'options' is a list of values or a single value
                option_values = values['options'] if isinstance(values['options'], list) else [values['options']]
                for option_value in option_values:
                    new_combo = {**combo_dict, cond_param: option_value}
                    yield new_combo
            else:
                # If conditions are not met, yield the base combination without the conditional parameter
                yield combo_dict


def format_config_name(config, config_basename, prefix, value_only, original_config):
    if value_only:
        config_items = [f"{v}" for _, v in config.items()]
    else:
        config_items = [f"{k}_{v}" for k, v in config.items()]

    return f"{prefix}{config_basename}-{'-'.join(config_items)}"

# Update the run_command function to handle list parameters correctly
def run_command(config, config_basename, output_dir, prefix, value_only, original_config):
    formatted_name = format_config_name(config, config_basename, prefix, value_only, original_config)
    config['tensorboard_run_name'] = formatted_name
    config['out_dir'] = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{formatted_name}")

    base_command = ["python3", "train.py"]
    for key, value in config.items():
        if isinstance(value, bool):
            base_command.extend([f"--{'' if value else 'no-'}{key}"])
        elif isinstance(value, list):
            # For list arguments, add each value separately
            for val in value:
                base_command.extend([f"--{key}", str(val)])
        else:
            base_command.extend([f"--{key}", str(value)])

    print(f"Running command: {' '.join(base_command)}")

    subprocess.run(base_command)

def main():
    args = parse_args()

    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    with open(args.config, 'r') as file:
        original_configurations = json.load(file)

    for config in original_configurations:
        for combination in generate_combinations(config):
            run_command(combination, config_basename, args.output_dir, args.prefix, args.value_only, original_configurations[0])

if __name__ == "__main__":
    main()


