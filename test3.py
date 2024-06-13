import json
import os
import sys
import pandas as pd
import argparse
from datetime import datetime
from itertools import product
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

def expand_range(value):
    if isinstance(value, dict) and 'range' in value:
        range_def = value['range']
        start, end = range_def['start'], range_def['end']
        step = range_def.get('step', 1 if isinstance(start, int) else 0.1)
        return list(range(start, end + 1, step)) if isinstance(start, int) else [start + i * step for i in range(int((end - start) / step) + 1)]
    return value

def generate_combinations(config):
    print("Generating parameter combinations...")
    parameter_groups = config.pop('parameter_groups', [{}])
    base_params = {k: expand_range(v) if isinstance(v, dict) and 'range' in v else v for k, v in config.items() if not isinstance(v, dict) or ('range' in v)}
    base_params = {k: v if isinstance(v, list) else [v] for k, v in base_params.items()}
    conditional_params = {k: v for k, v in config.items() if isinstance(v, dict) and 'conditions' in v}

    for group in parameter_groups:
        current_base_params = {**base_params, **group}
        base_combinations = list(product(*[[(k, v) for v in values] for k, values in current_base_params.items()]))

        for base_combination in base_combinations:
            combo_dict = dict(base_combination)
            valid_combos = [combo_dict]

            for cond_param, values in conditional_params.items():
                new_combos = []
                for combo in valid_combos:
                    if check_conditions(values['conditions'], combo):
                        option_values = values['options'] if isinstance(values['options'], list) else [values['options']]
                        for option_value in option_values:
                            new_combo = {**combo, cond_param: option_value}
                            new_combos.append(new_combo)
                    else:
                        new_combos.append(combo)
                valid_combos = new_combos

            for combo in valid_combos:
                yield combo

def format_config_name(config, config_basename, prefix, add_names):
    if add_names:
        config_items = [f"{k}_{v}" for k, v in config.items()]
    else:
        config_items = [f"{v}" for _, v in config.items()]

    return f"{prefix}{config_basename}-{'-'.join(config_items)}"

def run_experiment_with_vizier(config, config_basename, output_dir, csv_ckpt_dir, prefix, add_names, best_val_loss_from, override_max_iters, override_dataset, override_block_size):
    study_config = vz.StudyConfig.from_problem(
        vz.ProblemStatement(
            search_space={
                k: vz.ParameterConfig(k, parameter_type=type(v[0]).__name__.upper(), feasible_values=v)
                for k, v in config.items()
            },
            metric_information=[
                vz.MetricInformation(name="loss", goal=vz.ObjectiveMetricGoal.MINIMIZE)
            ]
        )
    )
    study_config.algorithm = "RANDOM_SEARCH"
    study_client = clients.Study.from_study_config(study_config, owner='owner', study_id='example_study_id')

    def evaluate(params):
        # Placeholder function to execute the experiment
        # You'll need to implement the actual evaluation logic
        # For now, we return a random value as the loss
        import random
        loss = random.random()
        return loss

    for _ in range(100):  # Replace with the number of iterations you want
        suggestions = study_client.suggest(count=1)
        for suggestion in suggestions:
            loss = evaluate(suggestion.parameters)
            suggestion.complete(vz.Measurement(metrics={'loss': loss}))

    best_trial = study_client.optimal_trials()[0].materialize()
    print(f"Best trial: {best_trial.parameters}, Loss: {best_trial.final_measurement.metrics['loss']}")

def main():
    args = parse_args()
    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    with open(args.config, 'r') as file:
        original_configurations = json.load(file)

    for config in original_configurations:
        for combination in generate_combinations(config):
            run_experiment_with_vizier(
                combination,
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

