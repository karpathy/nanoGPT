import torch, json, os, re
import wandb
import tiktoken

BEST_CONFIGS = [
    {"L": 100, "m": 5, "cache": True},
    {"L": 256, "m": 5, "cache": False},
    {"L": 512, "m": 10, "cache": True},
]

TASK = "gsm8k_answer_only"
DATASET_DIR = "data/gsm8k_answer_only"
DEVICE = "cuda"

import sys
sys.path.insert(0, "/kaggle/working/nanoGPT")
from model import GPT, SoftPrefix

api = wandb.Api()
ENTITY = api.default_entity
PROJECT = "nanoGPT-dissertation"
print(f"WandB entity: {ENTITY}")

DROPOUT = 0.0
MODEL_TYPE = "gpt2"
MAX_POSITIONS = 1024
MAX_NEW_TOKENS = 32

enc = tiktoken.get_encoding("gpt2")


def extract_number(text):
    if "####" in text:
        after = text.split("####")[-1].strip().replace(",", "")
        match = re.search(r"-?\d+\.?\d*", after)
        if match:
            return match.group()

    text_clean = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text_clean)
    if numbers:
        return numbers[-1]
    return None


def normalize_number(x):
    if x is None:
        return None
    try:
        value = float(str(x).replace(",", "").strip())
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return str(x).strip()


def download_artifact(L, m, cache, task):
    if L == 0:
        return "baseline"

    cache_tag = "cacheon" if cache else "cacheoff"
    artifact_name = f"{ENTITY}/{PROJECT}/prefix-L{L}-m{m}-{cache_tag}-{task}:latest"

    try:
        artifact = api.artifact(artifact_name, type="model")
        local_dir = artifact.download()
        print(f"  Downloaded: {artifact_name} -> {local_dir}")
        return local_dir
    except Exception as e:
        print(f"  Could not download artifact for L={L}, m={m}: {e}")
        return None


def load_checkpoint(L, m, cache, task):
    if L == 0:
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

    local_dir = download_artifact(L, m, cache, task)
    if local_dir is None:
        return None, None, None

    prefix_path = os.path.join(local_dir, "prefix_P.pt")
    if not os.path.exists(prefix_path):
        print(f"  No prefix_P.pt found for L={L}, m={m}")
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


def get_training_metrics(L, m, cache, task):
    cache_tag = "cacheon" if cache else "cacheoff"
    run_name = f"prefix-L{L}-m{m}-{cache_tag}-{task}"

    try:
        runs = api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={"display_name": run_name},
        )
        run = next(iter(runs))
        hist = run.history(keys=[
            "efficiency/tokens_per_sec",
            "efficiency/peak_gpu_mem_gb",
            "efficiency/wall_clock_sec",
            "prefix/cache_hit_ratio",
            "mfu",
            "val/em",
            "val/best_em",
        ])
        if hist.empty:
            return {}

        last = hist.iloc[-1]
        best_em = hist["val/em"].dropna().max() if "val/em" in hist else 0

        return {
            "tokens_per_sec": round(float(last.get("efficiency/tokens_per_sec", 0)), 1),
            "peak_mem_gb": round(float(last.get("efficiency/peak_gpu_mem_gb", 0)), 2),
            "wall_clock_sec": round(float(last.get("efficiency/wall_clock_sec", 0)), 1),
            "cache_hit_ratio": round(float(last.get("prefix/cache_hit_ratio", 0)), 3),
            "mfu": round(float(last.get("mfu", 0)), 2),
            "val_em": round(float(last.get("val/em", 0)), 4),
            "best_val_em": round(float(best_em), 4),
        }
    except Exception as e:
        print(f"  Could not fetch training metrics for L={L}, m={m}: {e}")
        return {}


def evaluate_em(model, soft_prefix, test_data, max_new_tokens=MAX_NEW_TOKENS, use_cache=False):
    model.eval()
    correct = 0
    total = 0

    prefix_kv = None
    if use_cache and soft_prefix is not None:
        with torch.no_grad():
            prefix_kv = model.build_prefix_kv(soft_prefix, batch_size=1)

    with torch.no_grad():
        for i, ex in enumerate(test_data):
            if i % 25 == 0:
                print(f"  test EM: {i}/{len(test_data)}", flush=True)

            prompt_ids = ex["input_ids"][:ex["prompt_len"]]
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

            if prefix_kv is not None:
                out = model.generate(idx, max_new_tokens, temperature=1.0, top_k=1, prefix_kv=prefix_kv)
            else:
                out = model.generate(idx, max_new_tokens, temperature=1.0, top_k=1, prefix=soft_prefix)

            generated_text = enc.decode(out[0].tolist())
            pred = normalize_number(extract_number(generated_text))
            gold = normalize_number(extract_number(ex["gold_answer"]))

            if pred is not None and gold is not None and pred == gold:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


test_data = torch.load(os.path.join(DATASET_DIR, "test.pt"))
print(f"Loaded {len(test_data)} answer-only test examples")

results = []

for cfg in BEST_CONFIGS:
    L = cfg["L"]
    M = cfg["m"]
    use_cache = cfg["cache"]

    if L >= MAX_POSITIONS:
        print(f"  SKIP: L={L} >= block_size={MAX_POSITIONS}")
        continue

    print(f"\nEvaluating L={L}, m={M}, cache={use_cache} on {TASK}...")
    model, soft_prefix, metadata = load_checkpoint(L, M, use_cache, TASK)
    if model is None:
        continue

    print(
        f"  model={metadata['model_type']} "
        f"block_size={metadata['block_size']} prefix_len={metadata['prefix_len']}"
    )

    em_score = evaluate_em(
        model,
        soft_prefix,
        test_data,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=use_cache,
    )
    param_count = metadata["prefix_len"] * model.config.n_embd

    r = {
        "L": metadata["prefix_len"],
        "m": M,
        "cache": use_cache,
        "test_em": round(em_score, 4),
        "param_count": param_count,
        "task": TASK,
        "model_type": metadata["model_type"],
        "block_size": metadata["block_size"],
    }
    r.update(get_training_metrics(L, M, use_cache, TASK))
    results.append(r)

    print(
        f"  L={L:5d}  test_EM={em_score:.4f}  "
        f"best_val_EM={r.get('best_val_em', 0):.4f}  "
        f"params={param_count:,}  tok/s={r.get('tokens_per_sec', '—')}"
    )

print("\n" + "═" * 80)
print(f"  H3 EVAL SUMMARY — task={TASK}")
print("═" * 80)
print(
    f"{'L':>6}  {'m':>4}  {'cache':>6}  {'test_EM':>8}  {'best_val':>8}  "
    f"{'params':>10}  {'tok/s':>8}  {'mem_gb':>7}  {'wall(s)':>8}"
)
print("-" * 90)

for r in results:
    print(
        f"{r['L']:>6}  {r['m']:>4}  {str(r['cache']):>6}  {r['test_em']:>8.4f}  "
        f"{r.get('best_val_em', 0):>8.4f}  {r['param_count']:>10,}  "
        f"{r.get('tokens_per_sec', 0):>8.1f}  "
        f"{r.get('peak_mem_gb', 0):>7.2f}  "
        f"{r.get('wall_clock_sec', 0):>8.1f}"
    )

eval_run = wandb.init(
    project=PROJECT,
    name=f"h3-eval-summary-{TASK}",
    job_type="eval",
)

wandb.log({
    "h3/summary_table": wandb.Table(
        columns=[
            "L", "m", "cache", "test_em", "best_val_em", "param_count",
            "model_type", "block_size",
            "tokens_per_sec", "peak_mem_gb", "wall_clock_sec",
            "cache_hit_ratio", "mfu",
        ],
        data=[[
            r["L"], r["m"], r["cache"], r["test_em"],
            r.get("best_val_em", 0), r["param_count"],
            r["model_type"], r["block_size"],
            r.get("tokens_per_sec", 0),
            r.get("peak_mem_gb", 0),
            r.get("wall_clock_sec", 0),
            r.get("cache_hit_ratio", 0),
            r.get("mfu", 0),
        ] for r in results],
    )
})
wandb.finish()

summary_path = "/kaggle/working/h3_gsm8k_answer_only_eval_summary.json"
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {summary_path}")