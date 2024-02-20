import json
import subprocess
import os
import sys
import pandas as pd
import argparse
from datetime import datetime
from itertools import product

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments based on a json configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file.")
    parser.add_argument("--output_dir", type=str, default="out", help="Directory to place the set of output checkpoints.")
    parser.add_argument("--csv_ckpt_dir", type=str, default="", help="Directory to place the set of csv checkpoints in csv_logs.")
    parser.add_argument("--prefix", type=str, default='', help="Optional prefix for tensorboard_run_name and out_dir.")
    parser.add_argument("--value_only", action="store_true", help="Include only the values of the configuration parameters in the names.")
    parser.add_argument("--use-best-val-loss-from", nargs=2, metavar=('csv_dir', 'output_dir'), type=str, default=['', ''],
                        help="Grab the best val loss of the run given by the csv_dir. Then, use the corresponding ckpt from the matching output_dir")
    return parser.parse_args()

def find_best_val_loss(csv_dir, output_dir):
    csvList = os.listdir(csv_dir)

    best_ckpt_loss = sys.float_info.max
    best_ckpt_name = ""
    for fname in csvList:
        ext = os.path.splitext(fname)
        params = ext[0].split('-')
        df = pd.read_csv(f"{csv_dir}/{fname}", header=None)
        best_valid_loss = df.iloc[:,-1].dropna().min()
        if best_valid_loss < best_ckpt_loss:
            best_ckpt_loss = best_valid_loss
            best_ckpt_name = ext[0]

    dirList = os.listdir(output_dir)
    for fname in dirList:
        params = best_ckpt_name.split('-', 2)
        if params[2] in fname:
            best_ckpt_name = fname
            break
        
    print("best_valid_loss: ", best_ckpt_loss)
    print("best_valid_chpt: ", best_ckpt_name)
    return f"{output_dir}/{best_ckpt_name}"

def check_conditions(conditions, combo_dict):
    return all(combo_dict.get(cond[0]) == cond[1] for cond in conditions)

def generate_combinations(config, current_combo={}, parent_conditions=[]):
    base_params = {k: v for k, v in config.items() if not isinstance(v, dict)}
    conditional_params = {k: v for k, v in config.items() if isinstance(v, dict)}

    if not conditional_params:  # If there are no conditional parameters
        yield from [dict(zip(base_params, combo)) for combo in product(*base_params.values())]
        return

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


def format_config_name(config, config_basename, prefix, value_only, original_config):
    if value_only:
        config_items = [f"{v}" for _, v in config.items()]
    else:
        config_items = [f"{k}_{v}" for k, v in config.items()]

    return f"{prefix}{config_basename}-{'-'.join(config_items)}"

def run_command(config, config_basename, output_dir, csv_ckpt_dir, prefix, value_only, best_val_loss_from, original_config):
    formatted_name = format_config_name(config, config_basename, prefix, value_only, original_config)
    config['tensorboard_run_name'] = formatted_name
    config['out_dir'] = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{formatted_name}")

    base_command = ["python3", "train.py"]
    for key, value in config.items():
        if isinstance(value, bool):
            base_command.extend([f"--{'' if value else 'no-'}{key}"])
        else:
            base_command.extend([f"--{key}", str(value)])

    if best_val_loss_from[0] and best_val_loss_from[1]:
        base_command.extend(["--init_from", "prev_run"])
        ckpt_path = find_best_val_loss(best_val_loss_from[0], best_val_loss_from[1])
        base_command.extend(["--prev_run_ckpt", ckpt_path])

    if csv_ckpt_dir:
        base_command.extend(["--csv_ckpt_dir", csv_ckpt_dir])

    print(f"Running command: {' '.join(base_command)}")
    subprocess.run(base_command)

def main():
    args = parse_args()

    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    with open(args.config, 'r') as file:
        original_configurations = json.load(file)

    for config in original_configurations:
        for combination in generate_combinations(config):
            run_command(combination, config_basename, args.output_dir, args.csv_ckpt_dir, args.prefix, 
                        args.value_only, args.use_best_val_loss_from, original_configurations[0])

if __name__ == "__main__":
    main()

