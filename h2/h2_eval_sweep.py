import time

import torch, math, json, os
import wandb
import datetime
# L_VALUES  = [0, 64, 100, 256, 512, 1024, 2048]
L_VALUES  = [0, 16, 32]
M_VALUES = [1, 5, 10, 20]
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
EVAL_WITH_CACHE = True
LR = 0.1 # established as best from LR sweep

def download_artifact(L, m, task):
    if L == 0:
        return "baseline"
    artifact_name = f"{ENTITY}/{PROJECT}/prefix-L{L}-m{m}-cacheon-{task}:latest"
    try:
        artifact = api.artifact(artifact_name, type="model")
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
        return model, None, {
            "model_type": MODEL_TYPE,
            "block_size": model.config.block_size,
            "prefix_len": 0,
            "prefix_update_period": 1,
        }

    local_dir = download_artifact(L, m, task)
    if local_dir is None:
        return None, None, None

    prefix_path = os.path.join(local_dir, "prefix_P.pt")
    if not os.path.exists(prefix_path):
        print(f"  No prefix_P.pt found for L={L}")
        return None, None, None

    saved = torch.load(prefix_path, map_location="cpu")
    saved_model_type = saved.get("model", MODEL_TYPE)
    saved_prefix_len = int(saved.get("prefix_len", L))
    saved_block_size = int(saved.get("block_size", MAX_POSITIONS - saved_prefix_len))
    saved_m = int(saved.get("prefix_update_period", m))

    model = GPT.from_pretrained(saved_model_type, dict(dropout=DROPOUT))
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    P_tensor = saved.get("P")
    if P_tensor is None:
        return model, None, {
            "model_type": saved_model_type,
            "block_size": saved_block_size,
            "prefix_len": saved_prefix_len,
            "prefix_update_period": saved_m,
        }

    if P_tensor.shape != (saved_prefix_len, model.config.n_embd):
        raise ValueError(
            f"Loaded prefix has shape {tuple(P_tensor.shape)}, expected "
            f"({saved_prefix_len}, {model.config.n_embd})"
        )

    soft_prefix = SoftPrefix(saved_prefix_len, model.config.n_embd, DEVICE)
    with torch.no_grad():
        soft_prefix.P.copy_(P_tensor.to(DEVICE))
    soft_prefix.to(DEVICE)
    soft_prefix.eval()

    return model, soft_prefix, {
        "model_type": saved_model_type,
        "block_size": saved_block_size,
        "prefix_len": saved_prefix_len,
        "prefix_update_period": saved_m,
    }


def get_training_metrics(L, m, task):
    run_name = f"prefix-L{L}-m{m}-cacheon-{task}"
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
                             token_block_size=None,
                             batch_size=6, eval_iters=50, use_cache=False):
    prefix_len = 0 if soft_prefix is None else soft_prefix.prefix_len
    max_token_block_size = model.config.block_size - prefix_len
    block_size = token_block_size or max_token_block_size
    block_size = min(block_size, max_token_block_size)
    assert block_size > 0, "prefix length must be smaller than the model block size"
    data       = np.memmap(data_path, dtype=np.uint16, mode='r')
    ctx        = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    losses     = []
    prefix_kv = None

    if use_cache and soft_prefix is not None:
        with torch.no_grad():
            prefix_kv = model.build_prefix_kv(soft_prefix, batch_size=batch_size)

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
                if prefix_kv is not None:
                    _, loss = model(X, Y, prefix=None, prefix_kv=prefix_kv)
                else:
                    _, loss = model(X, Y, prefix=soft_prefix)
            losses.append(loss.item())

    val_loss = float(np.mean(losses))
    return val_loss, math.exp(val_loss)


val_data = "data/wikitext2/val.bin"
results  = []

for L in L_VALUES:
    for M in M_VALUES:
            if L == 0 and M > M_VALUES[0]:
                continue

            print(f"\nEvaluating L={L}, m={M}, lr={LR}...")
            model, soft_prefix, metadata = load_checkpoint(L, M, TASK)
            if model is None:
                continue
            loaded_L = metadata["prefix_len"]
            loaded_m = metadata["prefix_update_period"]
            print(
                f"  model={metadata['model_type']} "
                f"block_size={metadata['block_size']} prefix_len={loaded_L} "
                f"train_m={loaded_m} eval_cache={int(EVAL_WITH_CACHE)}"
            )
            eval_start = time.time()
            val_loss, val_ppl = estimate_val_perplexity(
                model,
                soft_prefix,
                val_data,
                token_block_size=metadata["block_size"],
                use_cache=EVAL_WITH_CACHE,
            )
            eval_time = time.time() - eval_start
            param_count       = loaded_L * model.config.n_embd

            r = {
                "L":           loaded_L,
                "m":           M,
                "lr": LR,
                "val_loss":    round(val_loss, 4),
                "val_ppl":     round(val_ppl, 2),
                "param_count": param_count,
                "eval_time_sec": round(eval_time, 2),
                "task":        TASK,
                "model_type":  metadata["model_type"],
                "block_size":  metadata["block_size"],
                "eval_cache":  int(EVAL_WITH_CACHE),
            }
            r.update(get_training_metrics(L, M, TASK))
            results.append(r)

            print(f"  L={L:5d}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  "
                  f"params={param_count:,}  tok/s={r.get('tokens_per_sec', '—')}")


print("\n" + "═"*70)
print(f"  H2 EVAL SUMMARY — task={TASK}, m={M}")
print("═"*70)
print(f"{'L':>6}  {'val_ppl':>8}  {'params':>10}  "
      f"{'tok/s':>8}  {'mem_gb':>7}  {'wall(s)':>8}")
print("-"*70)
for r in results:
    print(f"{r['L']:>6}  {r['val_ppl']:>8.2f}  {r['param_count']:>10,}  "
          f"{r.get('tokens_per_sec', 0):>8.1f}  "
          f"{r.get('peak_mem_gb', 0):>7.2f}  "
          f"{r.get('wall_clock_sec', 0):>8.1f}")


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
eval_run = wandb.init(
    project  = PROJECT,
    name     = f"h2-eval-summary-{TASK}-cache-{EVAL_WITH_CACHE}-{timestamp}",
    job_type = "eval",
)
wandb.log({
    "h2/summary_table": wandb.Table(
        columns = [
            "L", "m", "val_loss", "val_ppl", "param_count",
            "model_type", "block_size", "eval_time_sec", "eval_cache",
            "tokens_per_sec", "peak_mem_gb", "wall_clock_sec",
            "cache_hit_ratio", "hit_ppl_50", "steps_to_target", "mfu",
        ],
        data = [[
            r["L"], r["m"], r["val_loss"], r["val_ppl"], r["param_count"],
            r["model_type"], r["block_size"], r["eval_time_sec"], r["eval_cache"],
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

with open("/kaggle/working/h2_eval_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to /kaggle/working/h2_eval_summary.json")
