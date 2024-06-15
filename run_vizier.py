import argparse
from datetime import datetime
import json
import os
import subprocess

from rich import print
from rich.console import Console
from rich.table import Table

import torch
from vizier.service import clients, pyvizier as vz


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vizier optimization based on json configuration file."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration JSON file."
    )

    parser.add_argument(
        "--add_names",
        action="store_true",
        help="Include names of values of the configuration parameters in addition to values (may cause too long a file name).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Directory to place the set of output checkpoints.",
    )

    parser.add_argument(
        "--vizier_iterations", type=int, default=20, help="Number of Vizier iterations."
    )

    parser.add_argument(
        "--vizier_algorithm",
        choices=[
            "GP_UCB_PE",
            "GAUSSIAN_PROCESS_BANDIT",
            "RANDOM_SEARCH",
            "QUASI_RANDOM_SEARCH",
            "GRID_SEARCH",
            "SHUFFLED_GRID_SEARCH",
            "EAGLE_STRATEGY",
            "CMA_ES",
            "EMUKIT_GP_EI",
            "NSGA2",
            "BOCS",
            "HARMONICA",
        ],
        default="GAUSSIAN_PROCESS_BANDIT",
        help="Choose the Vizier algorithm to use.",
    )

    return parser.parse_args()


def get_best_val_loss(out_dir):
    best_val_loss_file = out_dir + "/best_val_loss_and_iter.txt"
    if os.path.exists(best_val_loss_file):
        with open(best_val_loss_file, "r") as file:
            try:
                best_val_loss = float(file.readline().strip().split(",")[0])
                return best_val_loss
            except ValueError:
                print("val_loss file not found, trying checkpoint...")

    # if contained file doesn't exist, try ckpt.pt file
    checkpoint_file = out_dir + "/ckpt.pt"
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    best_val_loss = checkpoint["best_val_loss"]
    return best_val_loss


def format_config_name(config, config_basename, add_names):
    if add_names:
        config_items = [f"{k}_{v}" for k, v in config.items()]
    else:
        config_items = [f"{v}" for _, v in config.items()]

    return f"{config_basename}-{'-'.join(config_items)}"


def run_command(config, config_basename, output_dir, add_names):
    formatted_name = format_config_name(config, config_basename, add_names)
    base_command = ["python3", "train.py"]
    config["tensorboard_run_name"] = formatted_name
    timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["out_dir"] = os.path.join(output_dir, f"{timestamp_prefix}_{formatted_name}")
    base_command.extend(["--timestamp", timestamp_prefix])

    # Print the entered arguments before each run
    console = Console()
    table = Table(
        title="Entered Arguments", show_header=True, header_style="bold magenta"
    )
    table.add_column("Argument", style="cyan")
    table.add_column("Value", style="green")

    for key, value in config.items():
        table.add_row(key, str(value))

    console.print(table)

    # Create train.py command with argparse flags
    for key, value in config.items():
        if isinstance(value, bool):
            print(key, value, "bool")
            base_command.extend([f"--{'' if value else 'no-'}{key}"])
        elif value == "True":
            base_command.extend([f"--{key}"])
        elif value == "False":
            base_command.extend([f"--no-{key}"])
        elif isinstance(value, list):
            print(key, value, "list")
            for val in value:
                base_command.extend([f"--{key}", str(val)])
        else:
            print(key, value, "else")
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            base_command.extend([f"--{key}", str(value)])

    print(f"Running command: {' '.join(base_command)}")
    subprocess.run(base_command)
    return config


def run_experiment_with_vizier(
    config, config_basename, output_dir, add_names, vizier_algorithm, vizier_iterations
):
    search_space = vz.SearchSpace()
    for k, v in config.items():
        if isinstance(v, list):
            param_type = type(v[0]).__name__.upper()
            if param_type == "INT":
                search_space.root.add_int_param(
                    name=k, min_value=min(map(int, v)), max_value=max(map(int, v))
                )
            elif param_type == "FLOAT":
                search_space.root.add_float_param(
                    name=k, min_value=min(map(float, v)), max_value=max(map(float, v))
                )
            elif param_type == "STR":
                search_space.root.add_categorical_param(name=k, feasible_values=v)
            elif param_type == "BOOL":
                search_space.root.add_categorical_param(
                    name=k, feasible_values=[str(val) for val in v]
                )
        elif isinstance(v, dict) and "range" in v:
            range_def = v["range"]
            start, end, step = range_def["start"], range_def["end"], range_def["step"]
            param_type = type(start).__name__.upper()
            if param_type == "INT":
                search_space.root.add_int_param(
                    name=k,
                    min_value=start,
                    max_value=end,
                    scale_type=vz.ScaleType.LINEAR,
                )
            elif param_type == "FLOAT":
                search_space.root.add_float_param(
                    name=k,
                    min_value=start,
                    max_value=end,
                    scale_type=vz.ScaleType.LINEAR,
                )
        else:
            param_type = type(v).__name__.upper()
            if param_type == "INT":
                search_space.root.add_int_param(name=k, min_value=v, max_value=v)
            elif param_type == "FLOAT":
                search_space.root.add_float_param(name=k, min_value=v, max_value=v)
            elif param_type == "STR":
                search_space.root.add_categorical_param(name=k, feasible_values=[v])
            elif param_type == "BOOL":
                search_space.root.add_categorical_param(
                    name=k, feasible_values=[bool(v)]
                )

    print("search_space", search_space)
    study_config = vz.StudyConfig(
        search_space=search_space,
        metric_information=[
            vz.MetricInformation(name="loss", goal=vz.ObjectiveMetricGoal.MINIMIZE)
        ],
    )
    study_config.algorithm = vizier_algorithm
    study_client = clients.Study.from_study_config(
        study_config, owner="owner", study_id="example_study_id"
    )

    for i in range(vizier_iterations):
        print("Vizier Iteration", i)
        suggestions = study_client.suggest(count=1)
        for suggestion in suggestions:
            params = suggestion.parameters
            config = run_command(params, config_basename, output_dir, add_names)
            loss = get_best_val_loss(config["out_dir"])
            suggestion.complete(vz.Measurement(metrics={"loss": loss}))

    optimal_trials = study_client.optimal_trials()
    for trial in optimal_trials:
        best_trial = trial.materialize()
        print(
            f"Best trial: {best_trial.parameters}, Loss: {best_trial.final_measurement.metrics['loss']}"
        )


def main():
    args = parse_args()
    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    with open(args.config, "r") as file:
        original_configurations = json.load(file)

    for config in original_configurations:
        run_experiment_with_vizier(
            config,
            config_basename,
            args.output_dir,
            args.add_names,
            args.vizier_algorithm,
            args.vizier_iterations,
        )


if __name__ == "__main__":
    main()
