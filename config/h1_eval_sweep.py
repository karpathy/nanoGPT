import torch, math, json, os
import wandb

L_VALUES  = [0, 64, 100, 256, 512, 1024, 2048]
M_FIXED   = 1
TASK      = "wikitext2"
DEVICE    = "cuda"

import sys
sys.path.insert(0, "/kaggle/working/nanoGPT")
from model import GPT, GPTConfig, SoftPrefix
from contextlib import nullcontext
import numpy as np

def load_checkpoint(L, m, task):
    ckpt_path = f"/kaggle/working/h1_L{L}_m{m}/ckpt.pt"
    if not os.path.exists(ckpt_path):
        print(f"  Missing: {ckpt_path}")
        return None, None
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    gptconf = GPTConfig(**ckpt['model_args'])
    model   = GPT(gptconf)
    # strip DDP prefix if present
    state = {k.replace('_orig_mod.', ''): v
             for k, v in ckpt['model'].items()}
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)

    # load prefix P if it exists
    soft_prefix = None
    if L > 0 and ckpt.get('prefix_P') is not None:
        soft_prefix = SoftPrefix(
            prefix_len = L,
            n_layer    = gptconf.n_layer,
            n_head     = gptconf.n_head,
            n_embd     = gptconf.n_embd,
            device     = DEVICE,
        )
        soft_prefix.P = torch.nn.Parameter(ckpt['prefix_P'].to(DEVICE))
        soft_prefix.build_cache(model)

    return model, soft_prefix

def estimate_val_perplexity(model, soft_prefix, data_path, block_size=512,
                             batch_size=6, eval_iters=50):
    data  = np.memmap(data_path, dtype=np.uint16, mode='r')
    ctx   = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    losses = []

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

            prefix_kvs = soft_prefix.cached_kv if soft_prefix is not None else None
            with ctx:
                _, loss = model(X, Y, prefix_kvs=prefix_kvs)
            losses.append(loss.item())

    val_loss = float(np.mean(losses))
    return val_loss, math.exp(val_loss)

val_data = "data/wikitext2/val.bin"
results  = []

for L in L_VALUES:
    print(f"\nEvaluating L={L}, m={M_FIXED}...")
    model, soft_prefix = load_checkpoint(L, M_FIXED, TASK)
    if model is None:
        continue

    val_loss, val_ppl = estimate_val_perplexity(model, soft_prefix, val_data)
    param_count = L * 768

    results.append({
        "L":            L,
        "m":            M_FIXED,
        "val_loss":     round(val_loss, 4),
        "val_ppl":      round(val_ppl, 2),
        "param_count":  param_count,
        "task":         TASK,
    })
    print(f"  L={L:5d}  val_loss={val_loss:.4f}  "
          f"val_ppl={val_ppl:.2f}  params={param_count:,}")

print("\n" + "═"*60)
print(f"  H1 EVAL SUMMARY — task={TASK}, m={M_FIXED}")
print("═"*60)
print(f"{'L':>6}  {'val_loss':>9}  {'val_ppl':>8}  {'params':>10}")
print("-"*60)
for r in results:
    print(f"{r['L']:>6}  {r['val_loss']:>9.4f}  "
          f"{r['val_ppl']:>8.2f}  {r['param_count']:>10,}")

run = wandb.init(
    project = "nanoGPT-dissertation",
    name    = f"h1-eval-summary-{TASK}",
    job_type = "eval",
)
wandb.log({
    "h1/summary_table": wandb.Table(
        columns = ["L", "m", "val_loss", "val_ppl", "param_count"],
        data    = [[r["L"], r["m"], r["val_loss"],
                    r["val_ppl"], r["param_count"]]
                   for r in results]
    )
})
wandb.finish()

with open("/kaggle/working/h1_eval_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to /kaggle/working/h1_eval_summary.json")