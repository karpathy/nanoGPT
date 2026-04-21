#!/usr/bin/env python3
import argparse
import json
import re
# from datetime import UTC, datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a LaTeX summary table from structured experiment table data."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Optional JSON produced by collect_experiment_table_data.py.",
    )
    parser.add_argument(
        "--experiment-root",
        default="/scratch.global/chen8596/experiment_runs",
        help="Experiment output root used when --input is omitted.",
    )
    parser.add_argument("--output", default="", help="Optional output path for the LaTeX table")
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Family order to render. Defaults to GPT then LLAMA.",
    )
    parser.add_argument(
        "--column",
        action="append",
        default=[],
        help="Column spec in FAMILY:SIZE=LABEL form, e.g. GPT:124M=124M/5B",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method row label to render. Repeat to keep a fixed row order.",
    )
    parser.add_argument(
        "--method-label",
        action="append",
        default=[],
        help="Display label override in METHOD=LABEL form, e.g. cosine=Method1",
    )
    parser.add_argument(
        "--linesearch-label",
        default="linesearch_adam",
        help="Display label for the Adam line-search row.",
    )
    parser.add_argument(
        "--linesearch-muon-label",
        default="linesearch_muon",
        help="Display label for the Muon line-search row.",
    )
    parser.add_argument(
        "--loss-decimals",
        type=int,
        default=4,
        help="Number of decimals for loss values.",
    )
    parser.add_argument(
        "--rows-per-method",
        type=int,
        default=3,
        help="Number of data rows to render under each method block.",
    )
    parser.add_argument(
        "--size-label",
        action="append",
        default=[],
        help="Display label override in FAMILY:SIZE=LABEL form, e.g. GPT:124M=124M/5B",
    )
    return parser.parse_args()


def normalize_model_size(raw_size):
    raw_size = raw_size.strip()
    if raw_size and raw_size[-1].lower() in {"m", "b"}:
        return raw_size[:-1] + raw_size[-1].upper()
    return raw_size


def load_payload(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_columns(items):
    parsed = {}
    for item in items:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(
                f"Invalid --column value '{item}'. Expected FAMILY:SIZE=LABEL."
            )
        key, label = item.split("=", 1)
        family, size = key.split(":", 1)
        parsed.setdefault(family.strip().upper(), []).append(
            {"model_size": normalize_model_size(size), "label": label.strip()}
        )
    return parsed


def parse_method_labels(items):
    labels = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --method-label value '{item}'. Expected METHOD=LABEL."
            )
        method, label = item.split("=", 1)
        labels[method.strip()] = label.strip()
    return labels


def parse_size_labels(items):
    labels = {}
    for item in items:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(
                f"Invalid --size-label value '{item}'. Expected FAMILY:SIZE=LABEL."
            )
        key, label = item.split("=", 1)
        family, size = key.split(":", 1)
        labels[(family.strip().upper(), normalize_model_size(size))] = label.strip()
    return labels


def format_hours(value):
    if value is None:
        return ""
    value = float(value)
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return f"{rounded}h"
    return f"{value:.2f}h"


def format_loss(value, decimals):
    if value is None:
        return ""
    return f"{float(value):.{decimals}f}"


def discover_columns(entries, family_order):
    discovered = {family: [] for family in family_order}
    seen = {family: set() for family in family_order}
    for entry in entries:
        family = entry["family"].upper()
        if family not in discovered:
            continue
        key = entry["model_size"]
        if key in seen[family]:
            continue
        discovered[family].append(
            {"model_size": key, "label": entry.get("size_label", key)}
        )
        seen[family].add(key)
    for family in discovered:
        discovered[family].sort(key=lambda item: model_size_sort_key(item["model_size"]))
    return discovered


def build_entry_map(entries):
    entry_map = {}
    for entry in entries:
        key = (entry["family"].upper(), entry["model_size"], entry["method"])
        entry_map[key] = entry
    return entry_map


def build_candidate_map(entries):
    candidate_map = {}
    for entry in entries:
        key = (entry["family"].upper(), entry["model_size"], entry["method"])
        candidates = list(entry.get("candidates", []))
        candidates.sort(
            key=lambda item: (
                float("inf") if item.get("tuning_time_hours") is None else float(item["tuning_time_hours"]),
                float("inf") if item.get("loss") is None else float(item["loss"]),
            )
        )
        candidate_map[key] = candidates
    return candidate_map


def is_linesearch_method(method):
    return method in {"linesearch_adam", "linesearch_muon"}


def collect_body_methods(entries, explicit_methods):
    if explicit_methods:
        return [method for method in explicit_methods if not is_linesearch_method(method)]
    preferred = ["cosine", "muon", "schedulefree_adam"]
    methods = []
    for entry in entries:
        method = entry["method"]
        if is_linesearch_method(method):
            continue
        if method not in methods:
            methods.append(method)
    methods.sort(key=lambda method: (preferred.index(method) if method in preferred else len(preferred), method))
    return methods


def collect_linesearch_methods(entries, explicit_methods):
    if explicit_methods:
        methods = [method for method in explicit_methods if is_linesearch_method(method)]
    else:
        preferred = ["linesearch_adam", "linesearch_muon"]
        discovered = []
        for entry in entries:
            method = entry["method"]
            if is_linesearch_method(method) and method not in discovered:
                discovered.append(method)
        methods = [method for method in preferred if method in discovered]
        methods.extend([method for method in discovered if method not in methods])
    return methods


def build_tabular_spec(column_count):
    return "c|" + "|".join(["c|c|c"] * column_count)


def family_header_row(family_name, columns, total_columns):
    cells = [f"\\textbf{{{family_name}}}"]
    padded = list(columns) + [{"label": "Size"}] * (total_columns - len(columns))
    for index, column in enumerate(padded):
        suffix = "|" if index < total_columns - 1 else ""
        cells.append(f"\\multicolumn{{3}}{{c{suffix}}}{{{column['label']}}}")
    return " & ".join(cells) + " \\\\"


def method_header_row(method_label, total_columns):
    cells = [f"\\textbf{{{method_label}}}"]
    for _ in range(total_columns):
        cells.extend(["Tuning time", "Loss", "Spent time"])
    return " & ".join(cells) + " \\\\"


def method_data_row(method, family, columns, total_columns, candidate_map, row_index, loss_decimals):
    cells = [""]
    padded = list(columns) + [{"model_size": None}] * (total_columns - len(columns))
    for column in padded:
        if not column["model_size"]:
            cells.extend(["", "", ""])
            continue
        candidates = candidate_map.get((family, column["model_size"], method), [])
        if row_index >= len(candidates):
            cells.extend(["", "", ""])
            continue
        candidate = candidates[row_index]
        spent_time = candidate.get("total_spent_time_hours")
        if spent_time is None:
            spent_time = candidate.get("wall_clock_hours")
        cells.extend(
            [
                format_hours(candidate.get("tuning_time_hours")),
                format_loss(candidate.get("loss"), loss_decimals),
                format_hours(spent_time),
            ]
        )
    return " & ".join(cells) + " \\\\"


def linesearch_row(label, method, family, columns, total_columns, entry_map, loss_decimals):
    cells = [f"\\textbf{{{label}}}"]
    padded = list(columns) + [{"model_size": None}] * (total_columns - len(columns))
    for column in padded:
        if not column["model_size"]:
            cells.extend(["Nah", "", ""])
            continue
        entry = entry_map.get((family, column["model_size"], method))
        loss_value = ""
        spent_time_value = ""
        if entry:
            selected = entry["selected"]
            loss_value = format_loss(selected.get("loss"), loss_decimals)
            spent_time = selected.get("total_spent_time_hours")
            if spent_time is None:
                spent_time = selected.get("wall_clock_hours")
            spent_time_value = format_hours(spent_time)
        cells.extend(["Nah", loss_value, spent_time_value])
    return " & ".join(cells) + " \\\\"


def model_size_sort_key(model_size):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([MB])", model_size)
    if not match:
        return (float("inf"), model_size)
    number, suffix = match.groups()
    multiplier = 1_000 if suffix == "B" else 1
    return (float(number) * multiplier, model_size)


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


def collect_serial_halving_entries(experiment_root, size_labels):
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
        size_label = size_labels.get((family, model_size), model_size)
        method = infer_method(experiment_name, train_script)
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
                },
            )
        )
    return candidates


def collect_linesearch_entries(experiment_root, size_labels):
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
        size_label = size_labels.get((family, model_size), model_size)
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
                },
            )
        )
    return candidates


def aggregate_candidates(candidates):
    grouped = {}
    for candidate in candidates:
        key = (candidate["family"], candidate["model_size"], candidate["method"])
        grouped.setdefault(key, []).append(candidate)

    entries = []
    for key in sorted(grouped):
        rows = sorted(
            grouped[key],
            key=lambda item: (
                float("inf") if item.get("tuning_time_hours") is None else float(item["tuning_time_hours"]),
                float("inf") if item.get("loss") is None else float(item["loss"]),
                item["source_path"],
            ),
        )
        selected = min(
            rows,
            key=lambda item: (
                float("inf") if item.get("loss") is None else float(item["loss"]),
                float("inf") if item.get("tuning_time_hours") is None else float(item["tuning_time_hours"]),
                item["source_path"],
            ),
        )
        entries.append(
            {
                "family": key[0],
                "model_size": key[1],
                "size_label": selected["size_label"],
                "method": key[2],
                "selection_rule": "min_loss",
                "selected": selected,
                "candidates": rows,
            }
        )
    return entries


def build_payload_from_experiment_root(experiment_root, size_labels):
    candidates = []
    candidates.extend(collect_serial_halving_entries(experiment_root, size_labels))
    candidates.extend(collect_linesearch_entries(experiment_root, size_labels))
    return {
        # "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "experiment_root": str(experiment_root.resolve()),
        "entry_count": len(candidates),
        "entries": aggregate_candidates(candidates),
    }


def render_table(
    payload,
    family_order,
    explicit_columns,
    method_order,
    method_labels,
    loss_decimals,
    rows_per_method,
):
    entries = payload["entries"]
    if not entries:
        experiment_root = payload.get("experiment_root", "")
        hint = ""
        if experiment_root:
            hint = f" Input payload experiment_root={experiment_root!r}."
        raise ValueError(
            "No experiment entries found to render."
            + hint
            + " Re-run collect_experiment_table_data.py with the correct --experiment-root."
        )
    entry_map = build_entry_map(entries)
    candidate_map = build_candidate_map(entries)
    available_families = []
    if family_order:
        for family in family_order:
            if any(entry["family"].upper() == family for entry in entries):
                available_families.append(family)
    else:
        preferred = ["GPT", "LLAMA"]
        discovered = []
        for entry in entries:
            family = entry["family"].upper()
            if family not in discovered:
                discovered.append(family)
        available_families.extend([family for family in preferred if family in discovered])
        available_families.extend([family for family in discovered if family not in available_families])
    discovered_columns = discover_columns(entries, available_families)
    family_columns = {}
    for family in available_families:
        family_columns[family] = explicit_columns.get(family, discovered_columns.get(family, []))

    total_columns = max([len(columns) for columns in family_columns.values()] or [1])
    body_methods = collect_body_methods(entries, method_order)
    linesearch_methods = collect_linesearch_methods(entries, method_order)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{build_tabular_spec(total_columns)}}}")

    for family in available_families:
        lines.append("\\hline")
        lines.append(family_header_row(family, family_columns[family], total_columns))
        lines.append("\\hline")
        lines.append("")
        for method in body_methods:
            display_label = method_labels.get(method, method)
            lines.append(method_header_row(display_label, total_columns))
            lines.append("\\hline")
            for row_index in range(rows_per_method):
                lines.append(
                    method_data_row(
                        method=method,
                        family=family,
                        columns=family_columns[family],
                        total_columns=total_columns,
                        candidate_map=candidate_map,
                        row_index=row_index,
                        loss_decimals=loss_decimals,
                    )
                )
                lines.append("\\hline")
            lines.append("")
        for method in linesearch_methods:
            lines.append(
                linesearch_row(
                    label=method_labels.get(method, method),
                    method=method,
                    family=family,
                    columns=family_columns[family],
                    total_columns=total_columns,
                    entry_map=entry_map,
                    loss_decimals=loss_decimals,
                )
            )
            lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\caption{Comparison with tuning time, loss, and spent time}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    size_labels = parse_size_labels(args.size_label)
    if args.input:
        payload = load_payload(args.input)
    else:
        payload = build_payload_from_experiment_root(Path(args.experiment_root), size_labels)
    family_order = [item.upper() for item in args.family]
    explicit_columns = parse_columns(args.column)
    method_order = args.method
    method_labels = {
        "cosine": "cosine",
        "muon": "muon",
        "schedulefree_adam": "schedulefree_adam",
        "linesearch_adam": args.linesearch_label,
        "linesearch_muon": args.linesearch_muon_label,
    }
    method_labels.update(parse_method_labels(args.method_label))

    table_text = render_table(
        payload=payload,
        family_order=family_order,
        explicit_columns=explicit_columns,
        method_order=method_order,
        method_labels=method_labels,
        loss_decimals=args.loss_decimals,
        rows_per_method=args.rows_per_method,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table_text, encoding="utf-8")
    else:
        print(table_text)


if __name__ == "__main__":
    main()
