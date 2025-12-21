#!/usr/bin/env python3
"""
Plot training loss curves for nanoGPT baselines vs MEA.

Reads JSON files produced from `train.py` logs (see `plots/data/loss_curve_*.json`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def _ema(xs: list[float], beta: float) -> list[float]:
    if not xs:
        return []
    out: list[float] = []
    m = xs[0]
    for x in xs:
        m = beta * m + (1.0 - beta) * x
        out.append(m)
    return out


def _load(path: Path) -> tuple[list[int], list[float], dict, list[int], list[float]]:
    payload = json.loads(path.read_text())
    return (
        payload.get("iters", []),
        payload.get("loss", []),
        payload.get("meta", {}),
        payload.get("val_iters", []),
        payload.get("val_loss", []),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--softmax", type=str, default="plots/data/loss_curve_shakespeare_char_softmax.json")
    parser.add_argument("--mea", type=str, default="plots/data/loss_curve_shakespeare_char_mea.json")
    parser.add_argument("--out", type=str, default="plots/mea_loss_curve.png")
    parser.add_argument("--ema_beta", type=float, default=0.98)
    args = parser.parse_args()

    softmax_path = Path(args.softmax)
    mea_path = Path(args.mea)
    out_path = Path(args.out)

    it_s, loss_s, meta_s, vit_s, vloss_s = _load(softmax_path)
    it_m, loss_m, meta_m, vit_m, vloss_m = _load(mea_path)

    sns_style = "seaborn-v0_8-whitegrid"
    if sns_style in plt.style.available:
        plt.style.use(sns_style)

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(12.5, 5.6), constrained_layout=True)
    colors = {"softmax": "#111827", "mea": "#2563eb"}

    # Raw curves (light) + EMA-smoothed (bold)
    ax.plot(it_s, loss_s, color=colors["softmax"], alpha=0.15, lw=1.0)
    ax.plot(it_m, loss_m, color=colors["mea"], alpha=0.15, lw=1.0)

    ax.plot(it_s, _ema(loss_s, args.ema_beta), color=colors["softmax"], lw=2.4, label="Softmax (train, EMA)")
    ax.plot(it_m, _ema(loss_m, args.ema_beta), color=colors["mea"], lw=2.6, label="MEA (train, EMA)")

    if vit_s and vloss_s:
        ax.scatter(vit_s, vloss_s, s=26, marker="o", color=colors["softmax"], alpha=0.9, label="Softmax (val)")
    if vit_m and vloss_m:
        ax.scatter(vit_m, vloss_m, s=26, marker="o", color=colors["mea"], alpha=0.9, label="MEA (val)")

    ds = meta_s.get("dataset") or meta_m.get("dataset") or "dataset"
    max_it = max(it_s[-1] if it_s else 0, it_m[-1] if it_m else 0)
    ax.set_title(f"Loss curve: Softmax vs MEA ({ds}, {max_it} iters)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")

    # Annotate final training losses
    if it_s and loss_s:
        ax.annotate(
            f"{loss_s[-1]:.3f}",
            xy=(it_s[-1], loss_s[-1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0),
        )
    if it_m and loss_m:
        ax.annotate(
            f"{loss_m[-1]:.3f}",
            xy=(it_m[-1], loss_m[-1]),
            xytext=(10, -18),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
