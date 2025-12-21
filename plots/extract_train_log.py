#!/usr/bin/env python3
"""
Extract loss curves from a nanoGPT `train.py` stdout log.

Parses:
  - per-iter lines:
      iter 123: loss 3.1415, time 12.34ms, mfu 45.67%
  - eval lines:
      step 1000: train loss 3.1415, val loss 3.2718

Outputs a JSON artifact suitable for `plots/plot_loss_curve.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


_ITER_RE = re.compile(r"^iter (\d+): loss ([0-9.]+), time ([0-9.]+)ms, mfu ([0-9.]+)%")
_EVAL_RE = re.compile(r"^step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="path to captured stdout log")
    parser.add_argument("--out", type=str, required=True, help="output JSON path")
    parser.add_argument("--attn_type", type=str, default="", help="optional metadata")
    parser.add_argument("--dataset", type=str, default="", help="optional metadata")
    parser.add_argument("--config", type=str, default="", help="optional metadata")
    args = parser.parse_args()

    log_path = Path(args.log)
    text = log_path.read_text(errors="replace")

    # Use dicts to tolerate log concatenation / resume (duplicate iter numbers).
    it_by_iter: dict[int, float] = {}
    loss_by_iter: dict[int, float] = {}
    time_ms_by_iter: dict[int, float] = {}
    mfu_pct_by_iter: dict[int, float] = {}

    train_loss_eval_by_step: dict[int, float] = {}
    val_loss_by_step: dict[int, float] = {}

    for line in text.splitlines():
        line = line.strip()
        m = _ITER_RE.match(line)
        if m:
            it = int(m.group(1))
            it_by_iter[it] = it
            loss_by_iter[it] = float(m.group(2))
            time_ms_by_iter[it] = float(m.group(3))
            mfu_pct_by_iter[it] = float(m.group(4))
            continue
        m = _EVAL_RE.match(line)
        if m:
            step = int(m.group(1))
            train_loss_eval_by_step[step] = float(m.group(2))
            val_loss_by_step[step] = float(m.group(3))
            continue

    iters = sorted(it_by_iter)
    loss = [loss_by_iter[i] for i in iters]
    time_ms = [time_ms_by_iter[i] for i in iters]
    mfu_pct = [mfu_pct_by_iter[i] for i in iters]

    val_iters = sorted(val_loss_by_step)
    val_loss = [val_loss_by_step[i] for i in val_iters]
    train_loss_eval = [train_loss_eval_by_step[i] for i in val_iters]

    payload = {
        "meta": {
            "attn_type": args.attn_type,
            "dataset": args.dataset,
            "config": args.config,
            "created_unix": int(time.time()),
            "source_log": str(log_path),
        },
        "iters": iters,
        "loss": loss,
        "time_ms": time_ms,
        "mfu_pct": mfu_pct,
        "val_iters": val_iters,
        "val_loss": val_loss,
        "train_loss_eval": train_loss_eval,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path} (iters={len(iters)}, evals={len(val_iters)})")


if __name__ == "__main__":
    main()
