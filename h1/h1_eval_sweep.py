import torch, math, json, os
import wandb

# L_VALUES  = [0, 64, 100, 256, 512, 1024, 2048]
L_VALUES  = [512]
M_FIXED   = 1
TASK      = "wikitext2"
DEVICE    = "cuda"

import sys
sys.path.insert(0, "/kaggle/working/nanoGPT")
from model import GPT, GPTConfig, SoftPrefix
from contextlib import nullcontext
import numpy as np

api     = wandb.Api()
ENTITY  = api.default_entity
PROJECT = "nanoGPT-dissertation"
print(f"WandB entity: {ENTITY}")

DROPOUT = 0.0
MODEL_TYPE = 'gpt2'
MAX_POSITIONS = 1024

def download_artifact(L, m, task):
    if L == 0:
        return "baseline"  # no artifact needed for L=0
    artifact_name = f"{ENTITY}/{PROJECT}/prefix-L{L}-m{m}-{task}:latest"
    try:
        artifact  = api.artifact(artifact_name, type="model")
        local_dir = artifact.download()
        print(f"  Downloaded: {artifact_name} → {local_dir}")
        return local_dir
    except Exception as e:
        print(f"  Could not download artifact for L={L}: {e}")
        return None

def load_checkpoint(L, m, task):
    if L == 0:
        # baseline — just frozen GPT-2, no prefix
        model = GPT.from_pretrained(MODEL_TYPE, dict(dropout=DROPOUT))
        model.eval().to(DEVICE)
        for p in model.parameters():
            p.requires_grad = False
        return model, None

    local_dir = download_artifact(L, m, task)
    if local_dir is None:
        return None, None

    model = GPT.from_pretrained(MODEL_TYPE, dict(dropout=DROPOUT))
    if L > 0:
        effective_block_size = MAX_POSITIONS- L
        model.crop_block_size(effective_block_size)

    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    soft_prefix = None
    if L > 0:
        prefix_path = os.path.join(local_dir, "prefix_P.pt")
        if not os.path.exists(prefix_path):
            print(f"  No prefix_P.pt found for L={L}")
            return model, None

        saved = torch.load(prefix_path, map_location=DEVICE)
        P_tensor = saved['P']

        soft_prefix = SoftPrefix(L, model.config.n_embd, DEVICE)
        soft_prefix.P = torch.nn.Parameter(P_tensor.to(DEVICE))
        soft_prefix.to(DEVICE)

    return model, soft_prefix


def get_training_metrics(L, m, task):
    run_name = f"prefix-L{L}-m{m}-{task}"
    try:
        runs = api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={"display_name": run_name}
        )
        run  = next(iter(runs))
        hist = run.history(keys=[
            "efficiency/tokens_per_sec",
            "efficiency/peak_gpu_mem_gb",
            "efficiency/wall_clock_sec",
            "prefix/cache_hit_ratio",
            "target/hit_ppl_50",
            "target/steps_logged",
            "mfu",
        ])
        if hist.empty:
            return {}
        last = hist.iloc[-1]
        return {
            "tokens_per_sec":  round(float(last.get("efficiency/tokens_per_sec", 0)), 1),
            "peak_mem_gb":     round(float(last.get("efficiency/peak_gpu_mem_gb", 0)), 2),
            "wall_clock_sec":  round(float(last.get("efficiency/wall_clock_sec", 0)), 1),
            "cache_hit_ratio": round(float(last.get("prefix/cache_hit_ratio", 0)), 3),
            "hit_ppl_50":      int(last.get("target/hit_ppl_50", 0)),
            "steps_to_target": int(last.get("target/steps_logged", 0)),
            "mfu":             round(float(last.get("mfu", 0)), 2),
        }
    except Exception as e:
        print(f"  Could not fetch training metrics for L={L}: {e}")
        return {}


def estimate_val_perplexity(model, soft_prefix, data_path,
                             batch_size=6, eval_iters=50):
    block_size = model.config.block_size  # use whatever the model was trained with
    data       = np.memmap(data_path, dtype=np.uint16, mode='r')
    ctx        = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    losses     = []

    with torch.no_grad():
        for _ in range(eval_iters):
            ix = torch.randint(len(data) - block_size, (batch_size,))
            X  = torch.stack([
                torch.from_numpy(data[i:i+block_size].astype(np.int64))
                for i in ix
            ]).to(DEVICE)
            Y  = torch.stack([
                torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64))
                for i in ix
            ]).to(DEVICE)

            with ctx:
                _, loss = model(X, Y, prefix=soft_prefix)
            losses.append(loss.item())

    val_loss = float(np.mean(losses))
    return val_loss, math.exp(val_loss)


val_data = "data/wikitext2/val.bin"
results  = []

for L in L_VALUES:
    if L >= MAX_POSITIONS:
        print(f"  SKIP: L={L} — not trained, exceeds wpe table")
        continue

    print(f"\nEvaluating L={L}, m={M_FIXED}...")
    model, soft_prefix = load_checkpoint(L, M_FIXED, TASK)
    if model is None:
        continue
    print(f"  model.config.block_size={model.config.block_size}, prefix_len={L}")
    val_loss, val_ppl = estimate_val_perplexity(model, soft_prefix, val_data)
    param_count       = L * 768

    r = {
        "L":           L,
        "m":           M_FIXED,
        "val_loss":    round(val_loss, 4),
        "val_ppl":     round(val_ppl, 2),
        "param_count": param_count,
        "task":        TASK,
    }
    r.update(get_training_metrics(L, M_FIXED, TASK))
    results.append(r)

    print(f"  L={L:5d}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  "
          f"params={param_count:,}  tok/s={r.get('tokens_per_sec', '—')}")


print("\n" + "═"*70)
print(f"  H1 EVAL SUMMARY — task={TASK}, m={M_FIXED}")
print("═"*70)
print(f"{'L':>6}  {'val_ppl':>8}  {'params':>10}  "
      f"{'tok/s':>8}  {'mem_gb':>7}  {'wall(s)':>8}")
print("-"*70)
for r in results:
    print(f"{r['L']:>6}  {r['val_ppl']:>8.2f}  {r['param_count']:>10,}  "
          f"{r.get('tokens_per_sec', 0):>8.1f}  "
          f"{r.get('peak_mem_gb', 0):>7.2f}  "
          f"{r.get('wall_clock_sec', 0):>8.1f}")


eval_run = wandb.init(
    project  = PROJECT,
    name     = f"h1-eval-summary-{TASK}",
    job_type = "eval",
)
wandb.log({
    "h1/summary_table": wandb.Table(
        columns = [
            "L", "m", "val_loss", "val_ppl", "param_count",
            "tokens_per_sec", "peak_mem_gb", "wall_clock_sec",
            "cache_hit_ratio", "hit_ppl_50", "steps_to_target", "mfu",
        ],
        data = [[
            r["L"], r["m"], r["val_loss"], r["val_ppl"], r["param_count"],
            r.get("tokens_per_sec",  0),
            r.get("peak_mem_gb",     0),
            r.get("wall_clock_sec",  0),
            r.get("cache_hit_ratio", 0),
            r.get("hit_ppl_50",      0),
            r.get("steps_to_target", 0),
            r.get("mfu",             0),
        ] for r in results]
    )
})
wandb.finish()

with open("/kaggle/working/h1_eval_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to /kaggle/working/h1_eval_summary.json")