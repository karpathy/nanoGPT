import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import torch
import yaml

try:
    import optuna
except ImportError as exc:  # pragma: no cover - import guard for runtime
    raise SystemExit(
        "Optuna is required to run experiments. Install it first, for example with "
        "`pip install optuna` or by updating the project environment."
    ) from exc


TABLE_PATH = os.path.join("docs", "table.txt")
EVAL_LINE_RE = re.compile(
    r"^step (\d+): train loss "
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?), val loss "
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a fixed Optuna experiment.")
    parser.add_argument("config", help="Path to the experiment YAML config.")
    parser.add_argument(
        "--experiment-root",
        default="",
        help="Optional output directory. Defaults to output_root/name_timestamp.",
    )
    parser.add_argument(
        "--max-running-time-per-trial-hours",
        type=float,
        default=None,
        help="Override task.max_running_time_per_trial_hours for this stage-one run.",
    )
    parser.add_argument(
        "--num-iterations-per-trial",
        type=int,
        default=None,
        help="Override task.num_iterations_per_trial for this stage-one run.",
    )
    parser.add_argument(
        "--max-study-time-hours",
        type=float,
        default=None,
        help="Override optuna.max_study_time_hours for this stage-one run.",
    )
    parser.add_argument(
        "--stage1-result-path",
        default="",
        help="Optional path for a machine-readable stage-one result manifest.",
    )
    parser.add_argument(
        "--skip-runtime-table-validation",
        action="store_true",
        help="Validate experiment identity but allow runtime budget to differ from docs/table.txt.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a mapping.")
    return config


def require_keys(mapping, keys, section_name):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"Missing required keys in {section_name}: {missing}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def resolve_nproc_per_node(launch_config):
    raw_value = launch_config.get("nproc_per_node", "auto")
    if raw_value == "auto":
        gpu_count = torch.cuda.device_count()
        if gpu_count < 1:
            raise ValueError("launch.nproc_per_node=auto requires at least one visible CUDA device.")
        return gpu_count
    nproc = int(raw_value)
    if nproc < 1:
        raise ValueError("launch.nproc_per_node must be >= 1.")
    return nproc


def suggest_value(trial, name, spec):
    spec_type = spec["type"]
    if spec_type == "log_uniform":
        low, high = spec["range"]
        return trial.suggest_float(name, low, high, log=True)
    if spec_type == "uniform":
        low, high = spec["range"]
        return trial.suggest_float(name, low, high)
    if spec_type == "int":
        low, high = spec["range"]
        step = spec.get("step", 1)
        return trial.suggest_int(name, low, high, step=step)
    raise ValueError(f"Unsupported hyperparameter type for {name}: {spec_type}")


def parse_table(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if not parts or parts[0] == "family" or parts[0] == "---":
                continue
            if len(parts) != 8:
                continue
            rows.append(
                {
                    "family": parts[0],
                    "dataset": parts[1],
                    "model_size": parts[2],
                    "tokens_b": parts[3],
                    "train_target": parts[4],
                    "test_target": parts[5],
                    "runtime_hours": parts[6],
                    "notes": parts[7],
                }
            )
    return rows


def parse_optional_float(value):
    if value in {"NA", "", None}:
        return None
    return float(value)


def find_target_row(config):
    experiment = config["experiment"]
    table_rows = parse_table(TABLE_PATH)
    for row in table_rows:
        if (
            row["family"] == experiment["target_family"]
            and row["dataset"] == experiment["target_dataset"]
            and row["model_size"] == experiment["target_model_size"]
        ):
            return row
    raise ValueError(
        "Could not find matching target row in docs/table.txt for "
        f"{experiment['target_family']} / {experiment['target_dataset']} / {experiment['target_model_size']}."
    )


def validate_target_row(config, target_row, validate_runtime=True):
    task = config["task"]
    if "num_iterations_per_trial" in task:
        return
    expected_runtime = parse_optional_float(target_row["runtime_hours"])
    if (
        validate_runtime
        and expected_runtime is not None
        and float(task["max_running_time_per_trial_hours"]) != expected_runtime
    ):
        raise ValueError(
            "max_running_time_per_trial_hours mismatch: "
            f"config={task['max_running_time_per_trial_hours']} table={expected_runtime}"
        )


def build_pruner(config):
    optuna_cfg = config.get("optuna", {})
    pruning_cfg = optuna_cfg.get("pruning", {})
    if not pruning_cfg.get("enabled", True):
        return optuna.pruners.NopPruner()

    fixed_args = config.get("fixed_args", {})
    max_iters = fixed_args.get("max_iters")
    if max_iters is None:
        raise ValueError(
            "HyperbandPruner requires fixed_args.max_iters to compute resource budgets."
        )
    max_iters = float(max_iters)
    if max_iters <= 0:
        raise ValueError("fixed_args.max_iters must be > 0 for HyperbandPruner.")

    min_resource = max(1, int(0.01 * max_iters))
    return optuna.pruners.SuccessiveHalvingPruner(
        min_resource=min_resource,
        reduction_factor=4,
        min_early_stopping_rate=0,
    )


def choose_result_trial_from_optuna_trials(trials):
    completed_trials = [
        trial for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        return None, completed_trials
    result_trial = sorted(completed_trials, key=lambda trial: trial.number)[-1]
    return result_trial, completed_trials


def build_command(
    config,
    trial_dir,
    trial_id,
    sampled_params,
    summary_path,
    prune_signal_path,
    max_running_time_hours=None,
    num_iterations_override=None,
    init_from="scratch",
    stop_at_eval_boundary=False,
):
    experiment = config["experiment"]
    task = config["task"]
    checkpoint = config.get("checkpoint", {})
    fixed_args = config.get("fixed_args", {})
    launch = config.get("launch", {})
    num_iterations_per_trial = int(
        task["num_iterations_per_trial"]
        if num_iterations_override is None
        else num_iterations_override
    )

    launch_mode = launch.get("mode", "torchrun")
    if launch_mode == "torchrun":
        nproc_per_node = resolve_nproc_per_node(launch)
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
        ]
        master_port = launch.get("master_port")
        if master_port is not None:
            cmd.append(f"--master_port={int(master_port)}")
    elif launch_mode == "python":
        cmd = [sys.executable]
    else:
        raise ValueError(f"Unsupported launch.mode: {launch_mode}")

    cmd.append(experiment["train_script"])
    train_config = experiment.get("train_config", "")
    if train_config:
        cmd.append(train_config)
    for key, value in fixed_args.items():
        cmd.append(f"--{key}={value}")
    for key, value in sampled_params.items():
        cmd.append(f"--{key}={value}")
    cmd.extend(
        [
            f"--max_iters={num_iterations_per_trial}",
            f"--lr_decay_iters={task['num_iterations_per_trial']}",
            f"--out_dir={trial_dir}",
            f"--experiment_name={experiment['name']}",
            f"--trial_id={trial_id}",
            f"--experiment_metric_mode={task['metric_mode']}",
            "--max_running_time_hours=0.0",
            f"--save_last_checkpoint={checkpoint.get('save_last', False)}",
            f"--experiment_summary_path={summary_path}",
            f"--prune_signal_path={prune_signal_path}",
            f"--stop_at_eval_boundary={stop_at_eval_boundary}",
        ]
    )
    if init_from != "scratch":
        cmd.append(f"--init_from={init_from}")
    return cmd


def read_summary(summary_path):
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_learning_rate_from_optimizer_state(optimizer_state):
    if not isinstance(optimizer_state, dict):
        return None
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        return None
    first_group = param_groups[0]
    if not isinstance(first_group, dict) or "lr" not in first_group:
        return None
    return float(first_group["lr"])


def load_learning_rate_from_checkpoint(checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return extract_learning_rate_from_optimizer_state(checkpoint.get("optimizer"))


def load_learning_rate_from_run(summary=None, run_dir=""):
    candidate_paths = []
    if isinstance(summary, dict):
        candidate_paths.extend(
            [
                summary.get("best_checkpoint_path", ""),
                summary.get("last_checkpoint_path", ""),
            ]
        )
    if run_dir:
        candidate_paths.extend(
            [
                os.path.join(run_dir, "ckpt.pt"),
                os.path.join(run_dir, "ckpt_last.pt"),
            ]
        )

    seen = set()
    for path in candidate_paths:
        normalized = os.path.abspath(path) if path else ""
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        learning_rate = load_learning_rate_from_checkpoint(normalized)
        if learning_rate is not None:
            return learning_rate
    return None


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def get_metric_value(summary, metric_name):
    if metric_name == "val_loss":
        return float(summary["best_val_loss"])
    if metric_name == "train_loss":
        return float(summary["best_train_loss"])
    raise ValueError(f"Unsupported metric name: {metric_name}")


def write_study_summary(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def resolve_tuned_lr_param_name(hyperparameters):
    if len(hyperparameters) != 1:
        raise ValueError(
            "Experiment protocol allows tuning exactly one learning-rate hyperparameter. "
            f"Found hyperparameters: {sorted(hyperparameters.keys())}"
        )
    param_name = next(iter(hyperparameters.keys()))
    if param_name != "learning_rate" and not param_name.endswith("_lr"):
        raise ValueError(
            "Experiment protocol allows tuning only a learning-rate hyperparameter. "
            f"Found hyperparameter: {param_name!r}"
        )
    return param_name


def make_record(stage, trial_id, param_name, param_value, step, train_loss, val_loss, wall_clock_hours):
    record = {
        "stage": stage,
        "trial_id": trial_id,
        "step": int(step),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "wall_clock_hours": float(wall_clock_hours),
    }
    if param_name is not None:
        record["hyperparameter_name"] = param_name
    if param_value is not None:
        record["hyperparameter_value"] = float(param_value)
        record["learning_rate"] = float(param_value)
    return record


def stream_process(command, log_path, record_paths=None, record_context=None, trial=None, prune_signal_path=""):
    reported_steps = set()
    prune_requested = False
    process = subprocess.Popen(
        command,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    start_time = time.time()
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            match = EVAL_LINE_RE.match(line.strip())
            if not match:
                continue
            step = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            if step in reported_steps:
                continue
            reported_steps.add(step)
            wall_clock_hours = (time.time() - start_time) / 3600.0
            if record_paths is not None and record_context is not None:
                record_stage = record_context.get("stage", record_context.get("phase", "unknown"))
                record = make_record(
                    stage=record_stage,
                    trial_id=record_context["trial_id"],
                    param_name=record_context.get("hyperparameter_name"),
                    param_value=record_context.get("hyperparameter_value"),
                    step=step,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    wall_clock_hours=wall_clock_hours,
                )
                for path in record_paths:
                    append_jsonl(path, record)
            if trial is not None:
                trial.report(train_loss, step=step)
                if trial.should_prune() and not prune_requested:
                    with open(prune_signal_path, "w", encoding="utf-8") as f:
                        f.write("prune\n")
                    prune_requested = True
    return process.wait(), prune_requested


def main():
    start_time = time.time()
    if os.environ.get("RANK") not in (None, "0") or os.environ.get("WORLD_SIZE") not in (None, "1"):
        raise ValueError(
            "run_optuna_experiment.py must be launched as a single controller process. "
            "Do not use torchrun or multi-rank launchers for this script."
        )
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    slurm_procid = int(os.environ.get("SLURM_PROCID", "0"))
    if slurm_ntasks > 1 or slurm_procid > 0:
        raise ValueError(
            "run_optuna_experiment.py must be launched with a single SLURM task. "
            "Use one task and let it spawn torchrun for each trial."
        )

    args = parse_args()
    config = load_config(args.config)

    require_keys(config, ["experiment", "hyperparameters", "task"], "root")
    require_keys(
        config["experiment"],
        ["name", "train_script", "output_root", "target_family", "target_dataset", "target_model_size"],
        "experiment",
    )
    require_keys(
        config["task"],
        ["train_metric", "test_metric", "metric_mode", "num_iterations_per_trial"],
        "task",
    )

    hyperparameters = config["hyperparameters"]
    tuned_param_name = resolve_tuned_lr_param_name(hyperparameters)

    if args.max_running_time_per_trial_hours is not None:
        config["task"]["max_running_time_per_trial_hours"] = float(args.max_running_time_per_trial_hours)
    if args.num_iterations_per_trial is not None:
        config["task"]["num_iterations_per_trial"] = int(args.num_iterations_per_trial)

    experiment = config["experiment"]
    if not experiment.get("skip_table_validation", False):
        target_row = find_target_row(config)
        validate_target_row(
            config,
            target_row,
            validate_runtime=not args.skip_runtime_table_validation,
        )
    task = config["task"]
    checkpoint = config.setdefault("checkpoint", {})
    checkpoint["save_last"] = False
    launch = config.setdefault("launch", {})
    launch.setdefault("mode", "torchrun")
    launch.setdefault("nproc_per_node", "auto")
    optuna_cfg = config.setdefault("optuna", {})
    if args.max_study_time_hours is not None:
        max_study_time_hours = float(args.max_study_time_hours)
        optuna_cfg["max_study_time_hours"] = max_study_time_hours
    else:
        max_study_time_hours = float(optuna_cfg.get("max_study_time_hours", 0.0))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_root:
        experiment_root = ensure_dir(args.experiment_root)
    else:
        experiment_root = ensure_dir(
            os.path.join(experiment["output_root"], f"{experiment['name']}_{timestamp}")
        )
    trials_jsonl_path = os.path.join(experiment_root, "trials.jsonl")
    study_summary_path = os.path.join(experiment_root, "study_summary.json")
    all_records_path = os.path.join(experiment_root, "all_records.jsonl")
    config_snapshot_path = os.path.join(experiment_root, "resolved_config.yaml")
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    direction = "minimize" if task["metric_mode"] == "min" else "maximize"
    sampler = optuna.samplers.TPESampler()
    pruner = build_pruner(config)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    stop_reason = "study_timeout"

    def objective(trial):
        nonlocal stop_reason

        sampled_params = {
            name: suggest_value(trial, name, spec)
            for name, spec in hyperparameters.items()
        }
        trial_id = f"trial_{trial.number:04d}"
        trial_dir = ensure_dir(os.path.join(experiment_root, trial_id))
        summary_path = os.path.join(trial_dir, "summary.json")
        prune_signal_path = os.path.join(trial_dir, "PRUNE")
        log_path = os.path.join(trial_dir, "trial.log")
        records_path = os.path.join(trial_dir, "records.jsonl")
        if os.path.exists(prune_signal_path):
            os.remove(prune_signal_path)
        command = build_command(config, trial_dir, trial_id, sampled_params, summary_path, prune_signal_path)

        print(f"[experiment] starting {trial_id} with params {sampled_params}")
        returncode, prune_requested = stream_process(
            command,
            log_path,
            record_paths=[records_path, all_records_path],
            record_context={
                "stage": "stage1",
                "trial_id": trial_id,
                "hyperparameter_name": tuned_param_name,
                "hyperparameter_value": sampled_params[tuned_param_name],
            },
            trial=trial,
            prune_signal_path=prune_signal_path,
        )
        summary = read_summary(summary_path)
        if summary is None:
            failure_summary = {
                "trial_id": trial_id,
                "params": sampled_params,
                "returncode": returncode,
                "status": "missing_summary",
                "log_path": log_path,
            }
            append_jsonl(trials_jsonl_path, failure_summary)
            raise RuntimeError(
                f"Trial {trial_id} did not produce a summary file at {summary_path}."
            )

        train_objective_value = get_metric_value(summary, task["train_metric"])
        test_objective_value = get_metric_value(summary, task["test_metric"])
        pruned = bool(summary.get("termination_reason") == "pruned" or prune_requested)
        trial_record = {
            "trial_id": trial_id,
            "params": sampled_params,
            "selection_metric": task["train_metric"],
            "train_objective_value": train_objective_value,
            "test_objective_value": test_objective_value,
            "returncode": returncode,
            "summary_path": summary_path,
            "trial_dir": trial_dir,
            "log_path": log_path,
            "records_path": records_path,
            "termination_reason": summary.get("termination_reason", ""),
            "pruned": pruned,
        }
        append_jsonl(trials_jsonl_path, trial_record)

        for key, value in sampled_params.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("summary_path", summary_path)
        trial.set_user_attr("trial_dir", trial_dir)
        trial.set_user_attr("log_path", log_path)
        trial.set_user_attr("records_path", records_path)
        trial.set_user_attr("termination_reason", summary.get("termination_reason", ""))
        trial.set_user_attr("train_objective_value", train_objective_value)
        trial.set_user_attr("test_objective_value", test_objective_value)

        if returncode != 0:
            stop_reason = "trial_failed"
            raise RuntimeError(
                f"Trial {trial_id} failed with return code {returncode}."
            )

        if pruned:
            raise optuna.TrialPruned(f"Trial {trial_id} pruned.")

        return train_objective_value

    study.optimize(objective, timeout=max_study_time_hours * 3600.0)

    result_trial, completed_trials = choose_result_trial_from_optuna_trials(study.trials)
    pruned_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
    total_running_time_hours = max(0.0, (time.time() - start_time) / 3600.0)
    study_summary = {
        "experiment_name": experiment["name"],
        "experiment_root": experiment_root,
        "target_family": experiment["target_family"],
        "target_dataset": experiment["target_dataset"],
        "target_model_size": experiment["target_model_size"],
        "selection_metric": task["train_metric"],
        "tuned_hyperparameter_name": tuned_param_name,
        "best_trial_number": result_trial.number if result_trial is not None else None,
        "best_params": result_trial.params if result_trial is not None else None,
        "best_hyperparameter_value": (
            float(result_trial.params[tuned_param_name])
            if result_trial is not None and tuned_param_name in result_trial.params
            else None
        ),
        "best_learning_rate": (
            float(result_trial.params[tuned_param_name])
            if result_trial is not None and tuned_param_name in result_trial.params
            else None
        ),
        "best_value": result_trial.value if result_trial is not None else None,
        "best_train_value": result_trial.user_attrs.get("train_objective_value") if result_trial is not None else None,
        "best_test_value": result_trial.user_attrs.get("test_objective_value") if result_trial is not None else None,
        "selected_trial_id": (
            f"trial_{result_trial.number:04d}" if result_trial is not None else ""
        ),
        "selected_summary_path": result_trial.user_attrs.get("summary_path") if result_trial is not None else "",
        "selected_records_path": result_trial.user_attrs.get("records_path") if result_trial is not None else "",
        "selected_log_path": result_trial.user_attrs.get("log_path") if result_trial is not None else "",
        "selected_trial_dir": result_trial.user_attrs.get("trial_dir") if result_trial is not None else "",
        "direction": direction,
        "num_trials": len(study.trials),
        "num_completed_trials": len(completed_trials),
        "num_pruned_trials": len(pruned_trials),
        "num_iterations_per_trial": int(task["num_iterations_per_trial"]),
        "max_running_time_per_trial_hours": float(task["max_running_time_per_trial_hours"]),
        "max_study_time_hours": max_study_time_hours,
        "total_running_time_hours": total_running_time_hours,
        "all_records_path": os.path.abspath(all_records_path),
        "stop_reason": stop_reason,
    }
    write_study_summary(study_summary_path, study_summary)
    if args.stage1_result_path:
        stage1_result = {
            "schema_version": 1,
            "stage": "stage1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "total_running_time_hours": total_running_time_hours,
            "config_path": os.path.abspath(args.config),
            "config_snapshot_path": os.path.abspath(config_snapshot_path),
            "experiment_name": experiment["name"],
            "experiment_root": os.path.abspath(experiment_root),
            "stage1_root": os.path.abspath(experiment_root),
            "trials_jsonl_path": os.path.abspath(trials_jsonl_path),
            "study_summary_path": os.path.abspath(study_summary_path),
            "all_records_path": os.path.abspath(all_records_path),
            "max_running_time_per_trial_hours": float(task["max_running_time_per_trial_hours"]),
            "max_study_time_hours": float(max_study_time_hours),
            "selection_metric": task["train_metric"],
            "tuned_hyperparameter_name": tuned_param_name,
            "best_params": result_trial.params if result_trial is not None else None,
            "best_hyperparameter_value": (
                float(result_trial.params[tuned_param_name])
                if result_trial is not None and tuned_param_name in result_trial.params
                else None
            ),
            "best_learning_rate": (
                float(result_trial.params[tuned_param_name])
                if result_trial is not None and tuned_param_name in result_trial.params
                else None
            ),
            "best_trial_number": result_trial.number if result_trial is not None else None,
            "best_value": result_trial.value if result_trial is not None else None,
            "best_train_value": result_trial.user_attrs.get("train_objective_value") if result_trial is not None else None,
            "best_test_value": result_trial.user_attrs.get("test_objective_value") if result_trial is not None else None,
            "selected_trial_id": (
                f"trial_{result_trial.number:04d}" if result_trial is not None else ""
            ),
            "selected_summary_path": result_trial.user_attrs.get("summary_path") if result_trial is not None else "",
            "selected_records_path": result_trial.user_attrs.get("records_path") if result_trial is not None else "",
            "selected_log_path": result_trial.user_attrs.get("log_path") if result_trial is not None else "",
            "selected_trial_dir": result_trial.user_attrs.get("trial_dir") if result_trial is not None else "",
            "num_trials": len(study.trials),
            "num_completed_trials": len(completed_trials),
            "num_pruned_trials": len(pruned_trials),
            "num_iterations_per_trial": int(task["num_iterations_per_trial"]),
            "stop_reason": stop_reason,
        }
        stage1_result_dir = os.path.dirname(args.stage1_result_path)
        if stage1_result_dir:
            ensure_dir(stage1_result_dir)
        write_json(args.stage1_result_path, stage1_result)

    print("[experiment] completed")
    print(json.dumps(study_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
