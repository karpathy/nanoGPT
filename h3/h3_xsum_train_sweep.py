import subprocess, os, time, torch, json, math

BEST_CONFIGS = [
    {"L": 100, "m": 5, "cache": True},
    {"L": 256, "m": 5, "cache": False},
    {"L": 512, "m": 10, "cache": True},
]
TASK = "xsum"

results = []
training_errors = []
N_EMBD = 768
MAX_POSITIONS = 1024

for cfg in BEST_CONFIGS:
    L = cfg["L"]
    M = cfg["m"]
    cache_tag = "cacheon" if cfg["cache"] else "cacheoff"
    run_name = f"prefix-L{L}-m{M}-{cache_tag}-{TASK}"
    out_dir = f"/kaggle/working/{TASK}/h3_L{L}_m{M}_{cache_tag}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  RUN: L={L}  m={M}  task={TASK}")
    print(f"  out_dir:  {out_dir}")
    print(f"{'═'*60}")

    if L >= MAX_POSITIONS:
        print(f"  SKIP: L={L} >= block_size={MAX_POSITIONS}")
        results.append({"L": L, "m": M, "status": "SKIPPED", "wall_clock_h": 0})
        continue

    extra_flags = [f"--block_size={MAX_POSITIONS}"]
    if L == 0:
        extra_flags += [
            "--eval_only=True",
            "--always_save_checkpoint=True",
            "--eval_interval=1",
        ]
        print("  NOTE: L=0 baseline — eval only, no training")

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    cmd = [
        "python", "train.py", "config/h3_xsum.py",
        f"--prefix_len={L}",
        f"--prefix_update_period={M}",
        f"--run_final_em=False",
        f"--out_dir={out_dir}",
        f"--prefix_cache={cfg['cache']}",
        f"--prefix_type=soft",
        f"--wandb_run_name={run_name}",
        f"--block_size={MAX_POSITIONS}",
    ] + extra_flags

    proc = subprocess.run(cmd, capture_output=False, text=True)
    wall_clock = time.time() - t_start

    if proc.returncode != 0:
        print(f"  FAILED — returncode {proc.returncode}")
        training_errors.append((L, M))
        results.append({
            "L": L, "m": M, "status": "FAILED",
            "wall_clock_h": round(wall_clock / 3600, 2)
        })
        continue

    prefix_path = os.path.join(out_dir, "prefix_P.pt")
    if os.path.exists(prefix_path):
        ckpt = torch.load(prefix_path, map_location="cpu")
        val_loss = float(ckpt.get("val_loss", float("nan")))
        peak_mem = float(ckpt.get("peak_gpu_mem_gb", float("nan")))
        best_rougeL = float(ckpt.get("best_val_rougeL", float("nan")))

        results.append({
            "L": L,
            "m": M,
            "val_loss": round(val_loss, 4),
            "param_count": L * N_EMBD,
            "wall_clock_h": round(wall_clock / 3600, 2),
            "peak_mem_gb": round(peak_mem, 2),
            "best_rougeL": best_rougeL,
            "status": "OK",
        })
        print(f"  DONE — val_loss={val_loss:.4f}  best_rougeL={best_rougeL:.4f}  "
              f"time={wall_clock / 3600:.2f}h  mem={peak_mem:.2f}GB")
    else:
        print(f"  WARNING: no checkpoint at {prefix_path}")
        results.append({
            "L": L, "m": M, "status": "NO_CKPT",
            "wall_clock_h": round(wall_clock / 3600, 2)
        })

print(f"\n{'═'*60}")
print(f"  H3 {TASK} SWEEP COMPLETE")
print(f"{'═'*60}")
print(f"{'L':>6}  {'m':>4}  {'val_loss':>9}  {'best_RL':>9}  "
      f"{'params':>10}  {'mem(GB)':>7}  {'time(h)':>7}  status")
print("-"*78)
for r in results:
    if r["status"] == "OK":
        print(f"{r['L']:>6}  {r['m']:>4}  {r['val_loss']:>9.4f}  "
              f"{r['best_rougeL']:>9.4f}  "
              f"{r['param_count']:>10,}  {r['peak_mem_gb']:>7.2f}  "
              f"{r['wall_clock_h']:>7.2f}  OK")

    else:
        print(f"{r['L']:>6}  {r['m']:>4}  {'—':>9}  {'—':>10}  "
              f"{'—':>7}  {r['wall_clock_h']:>7.2f}  {r['status']}")

if training_errors:
    print(f"\nFailed (L, m) pairs: {training_errors}")

with open("/kaggle/working/h3_xsum_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: /kaggle/working/h3_xsum_summary.json")
