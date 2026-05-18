import subprocess, os, time, torch, json, math

# RUN_ORDER = [100, 0, 64, 256, 512, 1024, 2048]  # L=100 first — smoke test
RUN_ORDER = [256, 512, 760]  # L=100 first — smoke test
M_VALUES = [1, 5, 10, 20]

TASK      = "wikitext2"

results         = []
training_errors = []
N_EMBD = 768
MAX_POSITIONS = 1024
LR = 0.1 # established as best from LR sweep

# base_warmup = 200
# base_lr_decay = 5000
# base_eval_interval = 250

for L in RUN_ORDER:
    for M in M_VALUES:
            if L == 0 and M > M_VALUES[0]:
                continue
            run_name = f"prefix-L{L}-m{M}-{TASK}"
            out_dir = f"/kaggle/working/h2_L{L}_m{M}"
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n{'═'*60}")
            print(f"  RUN: L={L}  m={M}  task={TASK}")
            print(f"  out_dir:  {out_dir}")
            print(f"{'═'*60}")

            # skip L values that would make effective block_size <= 0
            if L >= MAX_POSITIONS:
                print(f"  SKIP: L={L} >= block_size={MAX_POSITIONS} — would give non-positive effective block_size")
                results.append({
                    "L": L, "status": "SKIPPED",
                    "wall_clock_h": 0
                })
                continue

            extra_flags = []
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
                "python", "train.py", "config/h1_wikitext2.py",
                f"--prefix_len={L}",
                f"--prefix_update_period={M}",
                f"--max_iters=2500",
                f"--learning_rate={LR}",
                f"--out_dir={out_dir}",
                f"--block_size={MAX_POSITIONS}",
                f"--prefix_type=soft",
                f"--wandb_run_name={run_name}",
                f"--prefix_cache=True",
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

            prefix_path = os.path.join(out_dir, "prefix_P.pt")
            if os.path.exists(prefix_path):
                ckpt     = torch.load(prefix_path, map_location="cpu")
                val_loss = float(ckpt.get("val_loss", float("nan")))
                val_ppl  = math.exp(val_loss) if not math.isnan(val_loss) else float("nan")
                peak_mem = float(ckpt.get("peak_gpu_mem_gb", 0.0))

                results.append({
                    "L":            L,
                    "m":            M,
                    "val_loss":     round(val_loss, 4),
                    "val_ppl":      round(val_ppl, 2),
                    "param_count":  L * N_EMBD,
                    "cache":        "on",
                    "wall_clock_h": round(wall_clock / 3600, 2),
                    "peak_mem_gb":  round(peak_mem, 2),
                    "status":       "OK",
                })
                print(f"  DONE — val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  "
                      f"time={wall_clock/3600:.2f}h  mem={peak_mem:.2f}GB")
            else:
                print(f"  WARNING: no checkpoint at {prefix_path}")
                results.append({
                    "L": L, "status": "NO_CKPT",
                    "wall_clock_h": round(wall_clock / 3600, 2)
                })

print(f"\n{'═'*60}")
print("  H2 TRAINING SWEEP COMPLETE")
print(f"{'═'*60}")
print(f"{'L':>6}  {'m':>4}  {'val_loss':>9}  {'val_ppl':>8}  "
      f"{'params':>10}  {'mem(GB)':>7}  {'time(h)':>7}  status")
print("-"*65)
for r in results:
    if r["status"] == "OK":
        print(f"{r['L']:>6}  {r['m']:>4}  {r['val_loss']:>9.4f}  {r['val_ppl']:>8.2f}  "
              f"{r['param_count']:>10,}  {r['peak_mem_gb']:>7.2f}  "
              f"{r['wall_clock_h']:>7.2f}  OK")
    else:
        print(f"{r['L']:>6}  {r.get('m', '—'):>4}  {'—':>9}  {'—':>8}  {'—':>10}  "
              f"{'—':>7}  {r['wall_clock_h']:>7.2f}  FAILED")

if training_errors:
    print(f"\nFailed L values: {training_errors}")

with open("/kaggle/working/h2_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: /kaggle/working/h2_summary.json")
