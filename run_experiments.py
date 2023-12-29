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
    parser.add_argument("--prefix", type=str, default='', help="Optional prefix for tensorboard_run_name and out_dir to help with grouping experiments.")
    return parser.parse_args()

def check_conditions(conditions, combo_dict):
    return all(combo_dict.get(cond[0]) == cond[1] for cond in conditions)

def generate_combinations(config, current_combo={}, parent_conditions=[]):
    base_params = {k: v for k, v in config.items() if not isinstance(v, dict)}
    conditional_params = {k: v for k, v in config.items() if isinstance(v, dict)}

    base_combinations = list(product(*[[(k, v) for v in (val if isinstance(val, list) else [val])] for k, val in base_params.items()]))

    for base_combination in base_combinations:
        combo_dict = {**current_combo, **dict(base_combination)}

        for cond_param, values in conditional_params.items():
            current_conditions = values['conditions'] + parent_conditions
            if check_conditions(current_conditions, combo_dict):
                for val in values['options']:
                    new_combo = {**combo_dict, cond_param: val}
                    if "nested" in values:
                        yield from generate_combinations(values["nested"], new_combo, current_conditions)
                    else:
                        yield new_combo
            else:
                yield combo_dict

def format_config_name(config, config_basename, prefix):
    config_items = [f"{k}_{v}" for k, v in config.items() if k != 'out_dir' and k != 'tensorboard_run_name']
    return f"{prefix}_{config_basename}_{'_'.join(config_items)}"

def run_command(config, config_basename, output_dir, prefix):
    formatted_name = format_config_name(config, config_basename, prefix)
    config['tensorboard_run_name'] = formatted_name
    config['out_dir'] = os.path.join(output_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{formatted_name}")

    base_command = ["python3", "train.py"]
    for key, value in config.items():
        if isinstance(value, bool):
            base_command.extend([f"--{'' if value else 'no-'}{key}"])
        else:
            base_command.extend([f"--{key}", str(value)])

    print(f"Running command: {' '.join(base_command)}")
    subprocess.run(base_command)

def main():
    args = parse_args()

    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    with open(args.config, 'r') as file:
        configurations = json.load(file)

    for config in configurations:
        for combination in generate_combinations(config):
            run_command(combination, config_basename, args.output_dir, args.prefix)

if __name__ == "__main__":
    main()

