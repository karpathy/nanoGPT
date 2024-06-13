import json
import torch
import subprocess
import os
import sys
import pandas as pd
import argparse
from datetime import datetime
from rich import print
from rich.console import Console
from rich.table import Table
from vizier.service import clients, pyvizier as vz

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments based on a json configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration JSON file.")
    parser.add_argument("--output_dir", type=str, default="out", help="Directory to place the set of output checkpoints.")
    parser.add_argument("--csv_ckpt_dir", type=str, default="", help="Directory to place the set of csv checkpoints in csv_logs.")
    parser.add_argument("--prefix", type=str, default='', help="Optional prefix for tensorboard_run_name and out_dir.")
    parser.add_argument("--add_names", action="store_true", help="Include names of values of the configuration parameters in addition to values (may cause too long a file name).")
    parser.add_argument("--use-best-val-loss-from", nargs=2, metavar=('csv_dir', 'output_dir'), type=str, default=['', ''],
                        help="Grab the best val loss of the run given by the csv_dir. Then, use the corresponding ckpt from the matching output_dir")
    parser.add_argument('--override_max_iters', default=None, type=int)
    parser.add_argument('--override_dataset', default=None, type=str)
    parser.add_argument('--override_block_size', default=None, type=int)
    return parser.parse_args()

def get_best_val_loss(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    best_val_loss = checkpoint['best_val_loss']
    return best_val_loss

def find_best_val_loss(csv_dir):
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

def format_config_name(config, config_basename, prefix, add_names):
    if add_names:
        config_items = [f"{k}_{v}" for k, v in config.items()]
    else:
        config_items = [f"{v}" for _, v in config.items()]

    return f"{prefix}{config_basename}-{'-'.join(config_items)}"

def run_command(config, config_basename, output_dir, csv_ckpt_dir, prefix, add_names,
                best_val_loss_from, override_max_iters, override_dataset, override_block_size):
    formatted_name = format_config_name(config, config_basename, prefix, add_names)
    base_command = ["python3", "train.py"]
    config['tensorboard_run_name'] = formatted_name
    timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['out_dir'] = os.path.join(output_dir, f"{timestamp_prefix}_{formatted_name}")
    base_command.extend(["--timestamp", timestamp_prefix])

    if override_max_iters:
        config['max_iters'] = str(override_max_iters)
    if override_dataset:
        config['dataset'] = override_dataset
    if override_block_size:
        config['block_size'] = str(override_block_size)

    # Print the entered arguments before each run
    console = Console()
    table = Table(title="Entered Arguments", show_header=True, header_style="bold magenta")
    table.add_column("Argument", style="cyan")
    table.add_column("Value", style="green")

    for key, value in config.items():
        table.add_row(key, str(value))

    console.print(table)

    for key, value in config.items():
        if isinstance(value, bool):
            base_command.extend([f"--{'' if value else 'no-'}{key}"])
        elif isinstance(value, list):
            for val in value:
                base_command.extend([f"--{key}", str(val)])
        else:
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            base_command.extend([f"--{key}", str(value)])

    if best_val_loss_from[0] and best_val_loss_from[1]:
        base_command.extend(["--init_from", "prev_run"])
        ckpt_path = find_best_val_loss(best_val_loss_from[0])
        base_command.extend(["--prev_run_ckpt", ckpt_path])

    if csv_ckpt_dir:
        base_command.extend(["--csv_ckpt_dir", csv_ckpt_dir])

    print(f"Running command: {' '.join(base_command)}")
    subprocess.run(base_command)
    return config

def run_experiment_with_vizier(config, config_basename, output_dir, csv_ckpt_dir, prefix, add_names, best_val_loss_from, override_max_iters, override_dataset, override_block_size):
    search_space = vz.SearchSpace()
    for k, v in config.items():
        if isinstance(v, list):
            param_type = type(v[0]).__name__.upper()
            if param_type == 'INT':
                search_space.root.add_int_param(name=k, min_value=min(map(int, v)), max_value=max(map(int, v)))
            elif param_type == 'FLOAT':
                search_space.root.add_float_param(name=k, min_value=min(map(float, v)), max_value=max(map(float, v)))
            elif param_type == 'STR':
                search_space.root.add_categorical_param(name=k, feasible_values=v)
        elif isinstance(v, dict) and 'range' in v:
            range_def = v['range']
            start, end, step = range_def['start'], range_def['end'], range_def['step']
            param_type = type(start).__name__.upper()
            if param_type == 'INT':
                search_space.root.add_int_param(name=k, min_value=start, max_value=end, scale_type=vz.ScaleType.LINEAR)
            elif param_type == 'FLOAT':
                search_space.root.add_float_param(name=k, min_value=start, max_value=end, scale_type=vz.ScaleType.LINEAR)
        else:
            param_type = type(v).__name__.upper()
            if param_type == 'INT':
                search_space.root.add_int_param(name=k, min_value=v, max_value=v)
            elif param_type == 'FLOAT':
                search_space.root.add_float_param(name=k, min_value=v, max_value=v)
            elif param_type == 'STR':
                search_space.root.add_categorical_param(name=k, feasible_values=[v])

    print("search_space", search_space)
    study_config = vz.StudyConfig(
        search_space=search_space,
        metric_information=[vz.MetricInformation(name="loss", goal=vz.ObjectiveMetricGoal.MINIMIZE)]
    )
    # study_config.algorithm = "RANDOM_SEARCH"
    study_config.algorithm = "GAUSSIAN_PROCESS_BANDIT"
    study_client = clients.Study.from_study_config(study_config, owner='owner', study_id='example_study_id')

    for i in range(100):  # Replace with the number of iterations you want
        print("Vizier Iteration", i)
        suggestions = study_client.suggest(count=1)
        for suggestion in suggestions:
            params = suggestion.parameters
            config = run_command(params, config_basename, output_dir, csv_ckpt_dir, prefix, add_names, best_val_loss_from, override_max_iters, override_dataset, override_block_size)
            loss = get_best_val_loss(config['out_dir'] + "/ckpt.pt")
            suggestion.complete(vz.Measurement(metrics={'loss': loss}))

    optimal_trials = study_client.optimal_trials()
    for trial in optimal_trials:
        best_trial = trial.materialize()
        print(f"Best trial: {best_trial.parameters}, Loss: {best_trial.final_measurement.metrics['loss']}")

def main():
    args = parse_args()
    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    with open(args.config, 'r') as file:
        original_configurations = json.load(file)

    for config in original_configurations:
        run_experiment_with_vizier(
            config,
            config_basename,
            args.output_dir,
            args.csv_ckpt_dir,
            args.prefix,
            args.add_names,
            args.use_best_val_loss_from,
            args.override_max_iters,
            args.override_dataset,
            args.override_block_size
        )

if __name__ == "__main__":
    main()

