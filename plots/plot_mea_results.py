#!/usr/bin/env python3
"""
Pretty plots for MEA benchmarks.

Reads JSON files produced by:
  - mea_sweep_attn.py
  - mea_train_smoke.py

Usage:
  nanogpt/plots/.venv/bin/python nanogpt/plots/plot_mea_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def _fmt_T(x, _pos=None):
    x = float(x)
    if x >= 1_000_000:
        return f"{x/1_000_000:.0f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{int(x)}"


def _fmt_ms(x, _pos=None):
    x = float(x)
    if x >= 1000:
        return f"{x/1000:.1f}s"
    return f"{x:.0f}ms"


def _load_sweep_json(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    df = pd.DataFrame(payload["rows"])
    df["T"] = df["T"].astype(int)
    return df.sort_values("T")


def _load_train_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _merge_sweeps(small: pd.DataFrame, large: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([small, large], ignore_index=True)
    df = df.sort_values("T")
    df = df.drop_duplicates(subset=["T"], keep="last")
    return df.reset_index(drop=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    out_dir = here

    fwd = _load_sweep_json(data_dir / "mea_fwd.json")
    fwd_bwd = _merge_sweeps(
        _load_sweep_json(data_dir / "mea_fwd_bwd_small.json"),
        _load_sweep_json(data_dir / "mea_fwd_bwd_large.json"),
    )
    decode_all = data_dir / "decode_all.json"
    if decode_all.exists():
        decode = _load_sweep_json(decode_all)
    else:
        decode = _merge_sweeps(
            _load_sweep_json(data_dir / "decode_small.json"),
            _load_sweep_json(data_dir / "decode_large.json"),
        )

    train_softmax = {
        262_144: _load_train_json(data_dir / "train_softmax_262k.json"),
        524_288: _load_train_json(data_dir / "train_softmax_524k.json"),
        1_048_576: _load_train_json(data_dir / "train_softmax_1m.json"),
    }
    train_mea = {
        262_144: _load_train_json(data_dir / "train_mea_262k.json"),
        524_288: _load_train_json(data_dir / "train_mea_524k.json"),
        1_048_576: _load_train_json(data_dir / "train_mea_1m.json"),
    }

    # ---- Styling (aim: presentation-ready)
    sns.set_theme(style="whitegrid", context="talk")
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "axes.titlepad": 10,
            "axes.labelpad": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#dddddd",
        }
    )

    palette = {
        "SDPA": "#2f2f2f",
        "MEA": "#2563eb",  # blue
    }

    # ---- Figure 1: attention-only dashboard
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95])
    ax_fwd = fig.add_subplot(gs[0, 0])
    ax_fwd_bwd = fig.add_subplot(gs[0, 1], sharex=ax_fwd)
    ax_speed = fig.add_subplot(gs[1, :], sharex=ax_fwd)

    for ax, df, title in (
        (ax_fwd, fwd, "Attention-only (forward)"),
        (ax_fwd_bwd, fwd_bwd, "Attention-only (forward + backward)"),
    ):
        ax.plot(df["T"], df["sdpa_ms"], marker="o", lw=2.5, color=palette["SDPA"], label="SDPA / FlashAttention")
        ax.plot(df["T"], df["mea_ms"], marker="o", lw=2.8, color=palette["MEA"], label="MEA (block + Triton)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("Time per iter (ms, log)")
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_ms))
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_T))
        ax.axhline(1000.0, color="#bbbbbb", lw=1.0, alpha=0.35)
        ax.legend(loc="upper left")

    # Speedup panel (SDPA / MEA)
    ax_speed.plot(fwd["T"], fwd["speedup"], marker="o", lw=2.8, color="#10b981", label="Speedup (fwd)")
    ax_speed.plot(fwd_bwd["T"], fwd_bwd["speedup"], marker="o", lw=2.8, color="#f59e0b", label="Speedup (fwd+bwd)")
    ax_speed.set_xscale("log", base=2)
    ax_speed.set_yscale("log")
    ax_speed.set_ylabel("SDPA / MEA speedup (log)")
    ax_speed.set_xlabel("Sequence length T")
    ax_speed.xaxis.set_major_formatter(FuncFormatter(_fmt_T))
    ax_speed.axhline(1.0, color="#111827", lw=1.3, alpha=0.5)
    ax_speed.axhline(10.0, color="#111827", lw=1.0, alpha=0.25, linestyle="--")
    ax_speed.fill_between(fwd_bwd["T"], 1.0, fwd_bwd["speedup"], where=fwd_bwd["speedup"] >= 1.0, color="#10b981", alpha=0.12)
    ax_speed.legend(loc="upper left", ncol=2)

    # Annotate a couple of big points
    def _annotate(ax, T, text, y):
        ax.annotate(
            text,
            xy=(T, y),
            xytext=(12, 10),
            textcoords="offset points",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#666666", lw=1.1),
        )

    # fwd: show 10x+ and 30x+ region
    if 524_288 in set(fwd["T"]):
        y = float(fwd.loc[fwd["T"] == 524_288, "speedup"].iloc[0])
        _annotate(ax_speed, 524_288, f"{y:.1f}× @ 524k (fwd)", y)
    if 1_048_576 in set(fwd["T"]):
        y = float(fwd.loc[fwd["T"] == 1_048_576, "speedup"].iloc[0])
        _annotate(ax_speed, 1_048_576, f"{y:.1f}× @ 1M (fwd)", y)

    fig.suptitle(
        "MEA vs SDPA (FlashAttention): scaling at ultra-long context\n"
        "(A100-80GB, bf16, B=1, nh=12, hs=64, MEA order=2, impl=block, kernel=triton, chunk=4096)",
        fontsize=18,
        fontweight="bold",
    )

    out1 = out_dir / "mea_attention_scaling.png"
    fig.savefig(out1, bbox_inches="tight")

    # ---- Figure 2: end-to-end training smoke (tok/s + time + memory)
    rows = []
    for T, payload in train_softmax.items():
        rows.append(
            {
                "T": T,
                "attn": "softmax",
                "tok_s": payload["metrics"]["tok_per_s"],
                "step_ms": payload["metrics"]["step_ms"],
                "peak_mib": payload["metrics"]["peak_mib"],
            }
        )
    for T, payload in train_mea.items():
        rows.append(
            {
                "T": T,
                "attn": "mea",
                "tok_s": payload["metrics"]["tok_per_s"],
                "step_ms": payload["metrics"]["step_ms"],
                "peak_mib": payload["metrics"]["peak_mib"],
            }
        )
    train_df = pd.DataFrame(rows).sort_values(["T", "attn"])

    fig2, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    ax_tok, ax_time, ax_mem = axes
    colors = {"softmax": "#111827", "mea": "#2563eb"}

    for attn, label in (("softmax", "Softmax (SDPA)"), ("mea", "MEA (Triton)")):
        df = train_df[train_df["attn"] == attn]
        ax_tok.plot(df["T"], df["tok_s"], marker="o", lw=2.8, color=colors[attn], label=label)
        ax_time.plot(df["T"], df["step_ms"], marker="o", lw=2.8, color=colors[attn], label=label)
        ax_mem.plot(df["T"], df["peak_mib"], marker="o", lw=2.8, color=colors[attn], label=label)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_T))
        ax.grid(True, alpha=0.25)

    ax_tok.set_yscale("log")
    ax_tok.set_title("Training throughput")
    ax_tok.set_ylabel("Tokens / second (log)")
    ax_tok.set_xlabel("Sequence length T")

    ax_time.set_yscale("log")
    ax_time.set_title("Step time")
    ax_time.set_ylabel("Time per step (ms, log)")
    ax_time.set_xlabel("Sequence length T")
    ax_time.yaxis.set_major_formatter(FuncFormatter(_fmt_ms))

    ax_mem.set_yscale("log")
    ax_mem.set_title("Peak memory")
    ax_mem.set_ylabel("Max allocated (MiB, log)")
    ax_mem.set_xlabel("Sequence length T")

    handles, labels = ax_tok.get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig2.suptitle(
        "End-to-end long-context step (tiny GPT)\n"
        "(B=1, n_layer=2, n_head=4, n_embd=256, bf16; loss on final position logits)",
        fontsize=18,
        fontweight="bold",
    )

    out2 = out_dir / "mea_train_smoke_scaling.png"
    fig2.savefig(out2, bbox_inches="tight")

    # ---- Figure 3: decode-time (KV-cache SDPA vs stateful MEA)
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    ax_tok, ax_ms, ax_mem, ax_speed = axes3.flatten()

    ax_tok.plot(decode["T"], decode["softmax_tok_s"], marker="o", lw=2.8, color=palette["SDPA"], label="SDPA / FlashAttention (KV-cache)")
    ax_tok.plot(decode["T"], decode["mea_tok_s"], marker="o", lw=2.8, color=palette["MEA"], label="MEA (stateful P/E)")
    ax_tok.set_xscale("log", base=2)
    ax_tok.set_yscale("log")
    ax_tok.set_title("Decode throughput")
    ax_tok.set_ylabel("Tokens / second (log)")
    ax_tok.set_xlabel("Context length T")
    ax_tok.xaxis.set_major_formatter(FuncFormatter(_fmt_T))
    ax_tok.legend(loc="upper right")

    ax_ms.plot(decode["T"], decode["softmax_ms_per_tok"], marker="o", lw=2.8, color=palette["SDPA"], label="SDPA / FlashAttention (KV-cache)")
    ax_ms.plot(decode["T"], decode["mea_ms_per_tok"], marker="o", lw=2.8, color=palette["MEA"], label="MEA (stateful P/E)")
    ax_ms.set_xscale("log", base=2)
    ax_ms.set_yscale("log")
    ax_ms.set_title("Decode latency")
    ax_ms.set_ylabel("Time per token (ms, log)")
    ax_ms.set_xlabel("Context length T")
    ax_ms.xaxis.set_major_formatter(FuncFormatter(_fmt_T))
    ax_ms.yaxis.set_major_formatter(FuncFormatter(_fmt_ms))

    ax_mem.plot(decode["T"], decode["softmax_kv_cache_mib"], marker="o", lw=2.8, color="#111827", label="Softmax KV cache (K+V)")
    ax_mem.plot(decode["T"], decode["mea_state_mib"], marker="o", lw=2.8, color="#10b981", label="MEA state (P+E)")
    ax_mem.set_xscale("log", base=2)
    ax_mem.set_yscale("log")
    ax_mem.set_title("Theoretical memory (per layer)")
    ax_mem.set_ylabel("MiB (log)")
    ax_mem.set_xlabel("Context length T")
    ax_mem.xaxis.set_major_formatter(FuncFormatter(_fmt_T))
    ax_mem.legend(loc="upper left")

    ax_speed.plot(decode["T"], decode["speedup"], marker="o", lw=2.8, color="#f59e0b", label="Speedup (SDPA / MEA)")
    ax_speed.set_xscale("log", base=2)
    ax_speed.set_yscale("log")
    ax_speed.set_title("Speedup")
    ax_speed.set_ylabel("SDPA / MEA (log)")
    ax_speed.set_xlabel("Context length T")
    ax_speed.xaxis.set_major_formatter(FuncFormatter(_fmt_T))
    ax_speed.axhline(1.0, color="#111827", lw=1.3, alpha=0.5)
    ax_speed.axhline(10.0, color="#111827", lw=1.0, alpha=0.25, linestyle="--")
    ax_speed.fill_between(decode["T"], 1.0, decode["speedup"], where=decode["speedup"] >= 1.0, color="#10b981", alpha=0.12)

    # Annotate a couple of big points
    for T_anno in (1_048_576, 2_097_152):
        if T_anno in set(decode["T"]):
            y = float(decode.loc[decode["T"] == T_anno, "speedup"].iloc[0])
            _annotate(ax_speed, T_anno, f"{y:.1f}× @ {_fmt_T(T_anno)} (decode)", y)

    fig3.suptitle(
        "Decode-time scaling: KV-cache SDPA vs stateful MEA\n"
        "(A100-80GB, bf16, B=1, nh=12, hs=64, MEA order=2; per-layer memory shown)",
        fontsize=18,
        fontweight="bold",
    )

    out3 = out_dir / "mea_decode_scaling.png"
    fig3.savefig(out3, bbox_inches="tight")

    print(f"wrote {out1}")
    print(f"wrote {out2}")
    print(f"wrote {out3}")


if __name__ == "__main__":
    main()
