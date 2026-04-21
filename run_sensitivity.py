"""Sensitivity analysis driver for llama60m.

Runs four independent parameter sweeps sequentially:
  - adamw_lr:           learning_rate   log_uniform[1e-5, 1e-2]
  - muon_lr:            muon_lr         log_uniform[1e-4, 1e-1]
  - linesearch_adam_c1: linesearch_c1   predefined list (see YAML)
  - linesearch_muon_c1: linesearch_c1   predefined list (see YAML)

A sweep uses a predefined list when its config has a `values:` key; otherwise
values are sampled from `distribution` over `range` with `num_trials` points.

Each trial is a single torchrun launch over 4 GPUs.
Intentionally independent from run_optuna_experiment / Optuna.

Usage:
    python run_sensitivity.py config/experiments/sensitivity_analysis.yaml
    python run_sensitivity.py config/experiments/sensitivity_analysis.yaml --only adamw_lr
    python run_sensitivity.py config/experiments/sensitivity_analysis.yaml --resume
    python run_sensitivity.py config/experiments/sensitivity_analysis.yaml --smoke
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml


LINESEARCH_SCRIPTS = {"train_linesearch_llama.py", "train_linesearch_llama_muon.py"}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trial_id_from_number(number):
    return f"trial_{number:04d}"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_values(distribution, lo, hi, n, rng):
    if distribution == "log_uniform":
        values = np.exp(rng.uniform(math.log(lo), math.log(hi), n))
    elif distribution == "uniform":
        values = rng.uniform(lo, hi, n)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    return sorted(float(v) for v in values)


def format_override(value):
    """Render a Python value for consumption by configurator.py's literal_eval."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported override value type: {type(value).__name__}")


def build_command(launch, train_script, train_config, overrides):
    mode = launch.get("mode", "torchrun")
    if mode == "torchrun":
        nproc = int(launch.get("nproc_per_node", 4))
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc}",
            train_script,
            train_config,
        ]
    elif mode == "python":
        cmd = [sys.executable, train_script, train_config]
    else:
        raise ValueError(f"Unsupported launch mode: {mode}")
    for key, value in overrides.items():
        cmd.append(f"--{key}={format_override(value)}")
    return cmd


def collect_summary_row(trial_dir, trial_id, param_name, param_value):
    summary_path = os.path.join(trial_dir, "summary.json")
    if not os.path.exists(summary_path):
        return {
            "trial_id": trial_id,
            param_name: param_value,
            "status": "missing_summary",
        }
    summary = read_json(summary_path)
    return {
        "trial_id": trial_id,
        param_name: param_value,
        "best_train_loss": summary.get("best_train_loss"),
        "best_val_loss": summary.get("best_val_loss"),
        "elapsed_wall_clock_hours": summary.get("elapsed_wall_clock_hours"),
        "termination_reason": summary.get("termination_reason"),
    }


def run_trial(cmd, trial_dir, trial_id):
    log_path = os.path.join(trial_dir, "trial.log")
    print(f"[{trial_id}] launching: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(
        f"[{trial_id}] exit={proc.returncode} in {elapsed/60:.1f} min  log={log_path}",
        flush=True,
    )
    return proc.returncode


def run_sweep(sweep, defaults, launch, output_root, rng, resume, smoke):
    name = sweep["name"]
    train_script = sweep["train_script"]
    train_config = sweep.get("train_config", defaults["train_config"])
    sweep_param = sweep["sweep_param"]

    fixed_overrides = dict(defaults.get("fixed_overrides") or {})
    fixed_overrides.update(sweep.get("extra_overrides") or {})

    if smoke:
        fixed_overrides["max_iters"] = 20
        fixed_overrides["lr_decay_iters"] = 20
        fixed_overrides["eval_interval"] = 10
        fixed_overrides["eval_iters"] = 2

    predefined = sweep.get("values")
    if predefined is not None:
        values = sorted(float(v) for v in predefined)
        if smoke:
            values = values[:2]
        num_trials = len(values)
        source_desc = f"predefined list n_trials={num_trials}"
    else:
        distribution = sweep["distribution"]
        lo, hi = sweep["range"]
        num_trials = int(sweep.get("num_trials", defaults["num_trials"]))
        if smoke:
            num_trials = 2
        values = sample_values(distribution, lo, hi, num_trials, rng)
        source_desc = f"dist={distribution} range=[{lo}, {hi}] n_trials={num_trials}"

    sweep_dir = ensure_dir(os.path.join(output_root, name))
    hyperparameters = {
        trial_id_from_number(i): {sweep_param: values[i]}
        for i in range(num_trials)
    }
    write_json(os.path.join(sweep_dir, "hyperparameters.json"), hyperparameters)

    print(f"\n=== sweep: {name} ===", flush=True)
    print(f"  param={sweep_param} {source_desc}", flush=True)
    print(f"  values: {values}", flush=True)

    want_records = train_script in LINESEARCH_SCRIPTS
    failures = []
    for i, value in enumerate(values):
        trial_id = trial_id_from_number(i)
        trial_dir = ensure_dir(os.path.join(sweep_dir, trial_id))
        summary_path = os.path.join(trial_dir, "summary.json")

        if resume and os.path.exists(summary_path):
            print(f"[{trial_id}] resume: summary.json exists, skipping", flush=True)
            continue

        overrides = {
            "out_dir": trial_dir,
            "experiment_name": f"{name}_sensitivity",
            "trial_id": trial_id,
            "experiment_summary_path": summary_path,
            sweep_param: value,
        }
        if want_records:
            overrides["experiment_records_path"] = os.path.join(trial_dir, "records.jsonl")
        overrides.update(fixed_overrides)

        resolved = {
            "trial_id": trial_id,
            "sweep": name,
            "sweep_param": sweep_param,
            "sweep_value": value,
            "train_script": train_script,
            "train_config": train_config,
            "overrides": overrides,
        }
        write_json(os.path.join(trial_dir, "resolved_config.json"), resolved)

        cmd = build_command(launch, train_script, train_config, overrides)
        returncode = run_trial(cmd, trial_dir, trial_id)
        if returncode != 0:
            failures.append({"trial_id": trial_id, "returncode": returncode})

    results = [
        collect_summary_row(
            os.path.join(sweep_dir, trial_id_from_number(i)),
            trial_id_from_number(i),
            sweep_param,
            values[i],
        )
        for i in range(num_trials)
    ]
    results.sort(key=lambda r: r[sweep_param])
    write_json(os.path.join(sweep_dir, "sweep_results.json"), results)

    if failures:
        print(f"[{name}] WARNING: {len(failures)} trial(s) failed: {failures}", flush=True)
    print(f"[{name}] aggregated results written to {sweep_dir}/sweep_results.json", flush=True)
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description="Run sensitivity-analysis sweeps for llama60m.")
    parser.add_argument("config", help="Path to the sensitivity YAML config.")
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Comma-separated sweep names to run (matches `sweeps[*].name`). "
            "Example: --only=adamw_lr,linesearch_adam_c1"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip trials whose summary.json already exists.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 2 trials with max_iters=20 per sweep. For plumbing validation only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_root = ensure_dir(cfg["output_root"])
    shutil.copy2(args.config, os.path.join(output_root, "sweep_config.yaml"))

    defaults = cfg.get("defaults", {}) or {}
    launch = cfg.get("launch", {}) or {}
    seed = int(cfg.get("seed", 42))

    all_sweeps = cfg["sweeps"]
    # Seed each sweep independently so that the sampled values for any one
    # sweep do not depend on which subset is run in this invocation. This lets
    # two nodes run different `--only` subsets and still produce coherent
    # results when combined.
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(len(all_sweeps))
    sweep_rngs = {
        s["name"]: np.random.default_rng(child_seeds[i])
        for i, s in enumerate(all_sweeps)
    }

    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        known = {s["name"] for s in all_sweeps}
        unknown = wanted - known
        if unknown:
            raise ValueError(
                f"--only contains unknown sweep name(s): {sorted(unknown)}. "
                f"Known sweeps: {sorted(known)}"
            )
        sweeps = [s for s in all_sweeps if s["name"] in wanted]
    else:
        sweeps = all_sweeps

    all_failures = {}
    for sweep in sweeps:
        failures = run_sweep(
            sweep=sweep,
            defaults=defaults,
            launch=launch,
            output_root=output_root,
            rng=sweep_rngs[sweep["name"]],
            resume=args.resume,
            smoke=args.smoke,
        )
        if failures:
            all_failures[sweep["name"]] = failures

    if all_failures:
        print("\nSome trials failed. Non-zero exit to surface the failure.", flush=True)
        print(json.dumps(all_failures, indent=2), flush=True)
        sys.exit(1)
    print("\nAll sweeps completed successfully.", flush=True)


if __name__ == "__main__":
    main()
