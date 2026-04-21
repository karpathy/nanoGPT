#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
# from datetime import UTC, datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect experiment summaries into a structured table dataset."
    )
    parser.add_argument(
        "--experiment-root",
        default="/scratch.global/chen8596/experiment_runs",
        help="Root directory containing experiment outputs.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--size-label",
        action="append",
        default=[],
        help="Override size display label, e.g. GPT:124M=124M/5B",
    )
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_model_size(raw_size):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([mMbB])", raw_size.strip())
    if not match:
        return raw_size.strip()
    number, suffix = match.groups()
    return f"{number}{suffix.upper()}"


def parse_size_overrides(items):
    overrides = {}
    for item in items:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(
                f"Invalid --size-label value '{item}'. Expected FAMILY:SIZE=LABEL."
            )
        key, label = item.split("=", 1)
        family, size = key.split(":", 1)
        overrides[(family.strip().upper(), normalize_model_size(size.strip()))] = label.strip()
    return overrides


def infer_family(experiment_name):
    lowered = experiment_name.lower()
    if lowered.startswith("gpt"):
        return "GPT"
    if lowered.startswith("llama"):
        return "LLAMA"
    return "UNKNOWN"


def infer_model_size(experiment_name):
    match = re.search(r"(?:gpt|llama)(\d+(?:\.\d+)?[mb])", experiment_name.lower())
    if not match:
        return "UNKNOWN"
    return normalize_model_size(match.group(1))


def infer_method(experiment_name, train_script):
    lowered_name = experiment_name.lower()
    lowered_script = (train_script or "").lower()
    if "line_search" in lowered_name or "linesearch" in lowered_name:
        if "muon" in lowered_name or "muon" in lowered_script:
            return "linesearch_muon"
        return "linesearch_adam"
    if "schedulefree" in lowered_name:
        return "schedulefree_adam"
    if "muon" in lowered_name or "muon" in lowered_script:
        return "muon"
    if "lr_search" in lowered_name or lowered_script == "train.py":
        return "cosine"
    return experiment_name


def is_linesearch_method(method):
    return method in {"linesearch_adam", "linesearch_muon"}


def compute_trial_total_spent_time_hours(trial_dir):
    records_path = Path(trial_dir) / "records.jsonl"
    if not records_path.exists():
        return None

    total = 0.0
    segment_max = None
    previous = None
    with records_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            current = record.get("wall_clock_hours")
            if current is None:
                continue
            current = float(current)
            if previous is not None and current < previous:
                if segment_max is not None:
                    total += segment_max
                segment_max = current
            else:
                segment_max = current if segment_max is None else max(segment_max, current)
            previous = current
    if segment_max is not None:
        total += segment_max
    return total if total > 0 else None


def make_candidate(
    family,
    model_size,
    size_label,
    method,
    tuning_time_hours,
    loss,
    wall_clock_hours,
    total_spent_time_hours,
    source_path,
    metadata,
):
    return {
        "family": family,
        "model_size": model_size,
        "size_label": size_label,
        "method": method,
        "tuning_time_hours": tuning_time_hours,
        "loss": loss,
        "wall_clock_hours": wall_clock_hours,
        "total_spent_time_hours": total_spent_time_hours,
        "source_path": str(source_path),
        "metadata": metadata,
    }


def collect_serial_halving_entries(experiment_root, size_overrides):
    candidates = []
    for result_path in sorted(experiment_root.glob("*/serial_halving_result.json")):
        payload = load_json(result_path)
        results = payload.get("results", [])
        if not results:
            continue
        result = results[-1]
        experiment_name = result.get("experiment_name", "")
        train_script = ""
        selected_summary_path = result.get("selected_summary_path")
        if selected_summary_path and Path(selected_summary_path).exists():
            selected_summary = load_json(selected_summary_path)
            train_script = selected_summary.get("train_script", "")
        family = (result.get("target_family") or infer_family(experiment_name)).upper()
        model_size = normalize_model_size(
            result.get("target_model_size") or infer_model_size(experiment_name)
        )
        method = infer_method(experiment_name, train_script)
        size_label = size_overrides.get((family, model_size), model_size)
        total_spent_time_hours = compute_trial_total_spent_time_hours(
            result.get("selected_trial_dir", "")
        )
        candidates.append(
            make_candidate(
                family=family,
                model_size=model_size,
                size_label=size_label,
                method=method,
                tuning_time_hours=result.get("total_running_time_hours"),
                loss=result.get("best_val_loss"),
                wall_clock_hours=result.get("elapsed_wall_clock_hours"),
                total_spent_time_hours=total_spent_time_hours,
                source_path=result_path,
                metadata={
                    "kind": "serial_halving",
                    "result_path": result.get("result_path", ""),
                    "rung_index": result.get("rung_index"),
                    "rung_name": result.get("rung_name", ""),
                    "num_trials": result.get("num_trials"),
                    "rung_target_iters": result.get("rung_target_iters"),
                },
            )
        )
    return candidates


def collect_linesearch_entries(experiment_root, size_overrides):
    candidates = []
    for summary_path in sorted(experiment_root.glob("*/final/summary.json")):
        if "/rung_" in str(summary_path):
            continue

        summary = load_json(summary_path)
        experiment_name = summary.get("experiment_name", "")
        method = infer_method(experiment_name, summary.get("train_script", ""))
        if not is_linesearch_method(method):
            continue

        family = infer_family(experiment_name).upper()
        model_size = infer_model_size(experiment_name)
        size_label = size_overrides.get((family, model_size), model_size)

        candidates.append(
            make_candidate(
                family=family,
                model_size=model_size,
                size_label=size_label,
                method=method,
                tuning_time_hours=None,
                loss=summary.get("best_val_loss"),
                wall_clock_hours=summary.get("elapsed_wall_clock_hours"),
                total_spent_time_hours=summary.get("elapsed_wall_clock_hours"),
                source_path=summary_path,
                metadata={
                    "kind": "linesearch_final",
                    "experiment_name": experiment_name,
                },
            )
        )
    return candidates


def aggregate_candidates(candidates):
    grouped = defaultdict(list)
    for candidate in candidates:
        key = (candidate["family"], candidate["model_size"], candidate["method"])
        grouped[key].append(candidate)

    entries = []
    for key in sorted(grouped):
        family, model_size, method = key
        rows = sorted(
            grouped[key],
            key=lambda item: (
                float("inf") if item["tuning_time_hours"] is None else item["tuning_time_hours"],
                float("inf") if item["loss"] is None else item["loss"],
                item["source_path"],
            ),
        )
        selected = min(
            rows,
            key=lambda item: (
                float("inf") if item["loss"] is None else item["loss"],
                float("inf") if item["tuning_time_hours"] is None else item["tuning_time_hours"],
                item["source_path"],
            ),
        )
        entries.append(
            {
                "family": family,
                "model_size": model_size,
                "size_label": selected["size_label"],
                "method": method,
                "selection_rule": "min_loss",
                "selected": selected,
                "candidates": rows,
            }
        )
    return entries


def main():
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    if not experiment_root.exists():
        raise FileNotFoundError(
            f"Experiment root does not exist: {experiment_root}"
        )
    size_overrides = parse_size_overrides(args.size_label)

    candidates = []
    candidates.extend(collect_serial_halving_entries(experiment_root, size_overrides))
    candidates.extend(collect_linesearch_entries(experiment_root, size_overrides))
    entries = aggregate_candidates(candidates)

    payload = {
        # "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "experiment_root": str(experiment_root.resolve()),
        "entry_count": len(entries),
        "entries": entries,
    }

    text = json.dumps(payload, indent=2, sort_keys=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
