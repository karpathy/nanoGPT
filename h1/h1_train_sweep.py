import subprocess, os, time, torch, json, math

RUN_ORDER = [100, 0, 64, 256, 512, 1024, 2048]  # L=100 first — smoke test
#RUN_ORDER = [100]  # L=100 first — smoke test
M_FIXED   = 1
TASK      = "wikitext2"

results         = []
training_errors = []

for L in RUN_ORDER:
    run_name = f"prefix-L{L}-m{M_FIXED}-{TASK}"
    out_dir  = f"/kaggle/working/h1_L{L}_m{M_FIXED}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  RUN: L={L}  m={M_FIXED}  task={TASK}")
    print(f"  out_dir:  {out_dir}")
    print(f"{'═'*60}")

    # L=2048 needs reduced batch to avoid OOM on 16GB T4
    extra_flags = []
    if L >= 2048:
        extra_flags = [
            "--batch_size=4",
            "--block_size=256",
            "--gradient_accumulation_steps=20",
        ]
        print("  NOTE: L=2048 — reduced batch_size=4, block_size=256")

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    cmd = [
        "python", "train.py", "config/h1_wikitext2.py",
        f"--prefix_len={L}",
        f"--prefix_update_period={M_FIXED}",
        f"--out_dir={out_dir}",
        f"--wandb_run_name={run_name}",
    ] + extra_flags

    proc = subprocess.run(cmd, capture_output=False, text=True)
    wall_clock = time.time() - t_start

    if proc.returncode != 0:
        print(f"  FAILED — returncode {proc.returncode}")
        training_errors.append(L)
        results.append({
            "L": L, "status": "FAILED",
            "wall_clock_h": round(wall_clock / 3600, 2)
        })
        continue

    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ckpt     = torch.load(ckpt_path, map_location="cpu")
        val_loss = float(ckpt.get("best_val_loss", float("nan")))
        val_ppl  = math.exp(val_loss) if not math.isnan(val_loss) else float("nan")
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        results.append({
            "L":            L,
            "m":            M_FIXED,
            "val_loss":     round(val_loss, 4),
            "val_ppl":      round(val_ppl, 2),
            "param_count":  L * 768,
            "wall_clock_h": round(wall_clock / 3600, 2),
            "peak_mem_gb":  round(peak_mem, 2),
            "status":       "OK",
        })
        print(f"  DONE — val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  "
              f"time={wall_clock/3600:.2f}h  mem={peak_mem:.2f}GB")
    else:
        print(f"  WARNING: no checkpoint at {ckpt_path}")

print(f"\n{'═'*60}")
print("  H1 TRAINING SWEEP COMPLETE")
print(f"{'═'*60}")
print(f"{'L':>6}  {'val_loss':>9}  {'val_ppl':>8}  "
      f"{'params':>10}  {'mem(GB)':>7}  {'time(h)':>7}  status")
print("-"*65)
for r in results:
    if r["status"] == "OK":
        print(f"{r['L']:>6}  {r['val_loss']:>9.4f}  {r['val_ppl']:>8.2f}  "
              f"{r['param_count']:>10,}  {r['peak_mem_gb']:>7.2f}  "
              f"{r['wall_clock_h']:>7.2f}  OK")
    else:
        print(f"{r['L']:>6}  {'—':>9}  {'—':>8}  {'—':>10}  "
              f"{'—':>7}  {r['wall_clock_h']:>7.2f}  FAILED")

if training_errors:
    print(f"\nFailed L values: {training_errors}")

with open("/kaggle/working/h1_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: /kaggle/working/h1_summary.json")