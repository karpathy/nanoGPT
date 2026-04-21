"""Aggregate sensitivity-analysis results and plot.

Reads the output produced by run_sensitivity.py and writes:
  - <output_root>/plots/<sweep>_sensitivity.png  (per-sweep figure)
  - <output_root>/plots/combined_2x2.png         (all four sweeps)
  - <output_root>/plots/summary.csv              (flat table of all trials)

Usage:
    python collect_sensitivity.py config/experiments/sensitivity_analysis.yaml
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_sweep_results(output_root, sweep_name):
    path = os.path.join(output_root, sweep_name, "sweep_results.json")
    if not os.path.exists(path):
        print(f"  [{sweep_name}] missing {path}, skipping")
        return None
    return read_json(path)


def usable_rows(rows, param_name):
    out = []
    for r in rows:
        if r.get("best_val_loss") is None:
            continue
        if r.get(param_name) is None:
            continue
        out.append(r)
    return out


def draw_panel(ax_loss, ax_time, rows, sweep_meta):
    name = sweep_meta["name"]
    param = sweep_meta["sweep_param"]
    distribution = sweep_meta.get("distribution")
    usable = usable_rows(rows, param)
    if not usable:
        ax_loss.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_loss.transAxes)
        ax_time.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_time.transAxes)
        ax_loss.set_title(name)
        return

    xs = [r[param] for r in usable]
    val_losses = [r["best_val_loss"] for r in usable]
    train_losses = [r.get("best_train_loss") for r in usable]
    hours = [r.get("elapsed_wall_clock_hours") for r in usable]

    ax_loss.plot(xs, val_losses, "o-", label="val loss", color="C0")
    if any(t is not None for t in train_losses):
        ax_loss.plot(xs, train_losses, "s--", label="train loss", color="C1", alpha=0.7)
    if distribution == "log_uniform":
        ax_loss.set_xscale("log")
    ax_loss.set_ylabel("best loss")
    ax_loss.set_title(f"{name}  (sweep on {param})")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="best")

    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
    ax_loss.annotate(
        f"min: {val_losses[best_idx]:.3f}\n@ {param}={xs[best_idx]:.3g}",
        xy=(xs[best_idx], val_losses[best_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
    )

    if any(h is not None for h in hours):
        hours_clean = [h if h is not None else float("nan") for h in hours]
        ax_time.plot(xs, hours_clean, "^-", color="C2")
    if distribution == "log_uniform":
        ax_time.set_xscale("log")
    ax_time.set_xlabel(param)
    ax_time.set_ylabel("wall-clock hours")
    ax_time.grid(True, alpha=0.3)


def make_single_figure(rows, sweep_meta, out_path):
    fig, (ax_loss, ax_time) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    draw_panel(ax_loss, ax_time, rows, sweep_meta)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_combined_figure(all_rows, sweeps_meta, out_path):
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), gridspec_kw={"width_ratios": [1, 1]})
    # Lay out 2x2 of (loss, time) panels -> use rows[i] for loss, rows[i] second col for time...
    # Simpler: 2 rows x 2 cols of sweeps, each has its own stacked loss/time via subgridspec.
    plt.close(fig)

    fig = plt.figure(figsize=(14, 12))
    outer = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    for i, meta in enumerate(sweeps_meta):
        r, c = divmod(i, 2)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
        ax_loss = fig.add_subplot(inner[0])
        ax_time = fig.add_subplot(inner[1], sharex=ax_loss)
        rows = all_rows.get(meta["name"]) or []
        draw_panel(ax_loss, ax_time, rows, meta)
        plt.setp(ax_loss.get_xticklabels(), visible=False)
    fig.suptitle("llama60m sensitivity analysis", fontsize=14)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_summary_csv(all_rows, sweeps_meta, out_path):
    fieldnames = [
        "sweep",
        "trial_id",
        "sweep_param",
        "sweep_value",
        "best_val_loss",
        "best_train_loss",
        "elapsed_wall_clock_hours",
        "termination_reason",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for meta in sweeps_meta:
            rows = all_rows.get(meta["name"]) or []
            for row in rows:
                writer.writerow(
                    {
                        "sweep": meta["name"],
                        "trial_id": row.get("trial_id"),
                        "sweep_param": meta["sweep_param"],
                        "sweep_value": row.get(meta["sweep_param"]),
                        "best_val_loss": row.get("best_val_loss"),
                        "best_train_loss": row.get("best_train_loss"),
                        "elapsed_wall_clock_hours": row.get("elapsed_wall_clock_hours"),
                        "termination_reason": row.get("termination_reason"),
                    }
                )
    print(f"  wrote {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate + plot sensitivity-analysis results.")
    parser.add_argument("config", help="Path to the sensitivity YAML config.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_root = cfg["output_root"]
    plots_dir = ensure_dir(os.path.join(output_root, "plots"))

    sweeps_meta = cfg["sweeps"]
    all_rows = {}
    for meta in sweeps_meta:
        name = meta["name"]
        rows = load_sweep_results(output_root, name)
        if rows is None:
            continue
        all_rows[name] = rows
        make_single_figure(
            rows=rows,
            sweep_meta=meta,
            out_path=os.path.join(plots_dir, f"{name}_sensitivity.png"),
        )

    make_combined_figure(all_rows, sweeps_meta, os.path.join(plots_dir, "combined_2x2.png"))
    write_summary_csv(all_rows, sweeps_meta, os.path.join(plots_dir, "summary.csv"))


if __name__ == "__main__":
    main()
