import datetime
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import wandb


RUN_IDS = [
    # cache on: L=760
    "wah5y719",  # m=20
    "hy19iy7g",  # m=10
    "efbe5ioq",  # m=5
    "18zt8g31",  # m=1

    # cache on: L=256
    "z5c80orx",  # m=20
    "w167iii8",  # m=10
    "tcb4odl2",  # m=5
    "g89cvglz", # m=1

    # cache on: L=100
    "i9chd58m", # m = 20
    "4snfh2xu", #m=10
    "gmzkhs0u", # m=5
    "eosffsx5", # m=1
    
    # cache on: L=64
    "szkvt3w0",  # m=20
    "bnvkjq07",  # m=10
    "rrrhqh8s",  # m=5
    "oc6cluzg", # m=1

    # cache on: L=32
    "j56obwel",  # m=20
    "7i12x9a1",  # m=10
    "sdc71ir5",  # m=5

   # cache on: L=16
    "khg3i8re", # m=20
    "n2izc5s6", # m=10
    "f2mw7od6", # m=5

    #cache on: L = 0
    "f2mw7od6"
]
PROJECT = "nanoGPT-dissertation"
TASK = "wikitext2"
DEVICE = "cuda"
MODEL_TYPE = "gpt2"
DROPOUT = 0.0
MAX_POSITIONS = 1024
VAL_DATA = "data/wikitext2/val.bin"

# "match" evaluates each run using its training cache mode.
# Set to True or False to force one eval mode for every run.
EVAL_CACHE_MODE = "match"
EVAL_BATCH_SIZE = 2
EVAL_ITERS = 50


sys.path.insert(0, "/kaggle/working/nanoGPT")
from model import GPT, SoftPrefix


api = wandb.Api()
ENTITY = api.default_entity
print(f"W&B entity: {ENTITY}")

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def parse_run_identity(run):
    match = re.search(
        r"prefix-L(?P<L>\d+)-m(?P<m>\d+)-cache(?P<cache>on|off)-",
        run.name,
    )
    if match:
        return (
            int(match.group("L")),
            int(match.group("m")),
            match.group("cache") == "on",
        )

    config = run.config
    if "prefix_len" not in config or "prefix_update_period" not in config:
        raise ValueError(
            f"Cannot infer L/m from run {run.id} ({run.name}); "
            "the name and config both lack the required fields"
        )

    return (
        int(config["prefix_len"]),
        int(config["prefix_update_period"]),
        bool(config.get("prefix_cache", False)),
    )


def find_prefix_file(directory):
    for root, _, files in os.walk(directory):
        if "prefix_P.pt" in files:
            return os.path.join(root, "prefix_P.pt")
    return None


def download_run_prefix(run, prefix_len):
    if prefix_len == 0:
        return None, None

    artifacts = [
        artifact
        for artifact in run.logged_artifacts()
        if artifact.type == "model"
    ]
    if not artifacts:
        raise FileNotFoundError(
            f"Run {run.id} ({run.name}) has no logged model artifact"
        )

    def version_number(artifact):
        match = re.search(r"v(\d+)$", artifact.version or "")
        return int(match.group(1)) if match else -1

    artifact = max(artifacts, key=version_number)
    local_dir = artifact.download()
    prefix_path = find_prefix_file(local_dir)
    if prefix_path is None:
        raise FileNotFoundError(
            f"Artifact {artifact.name} from run {run.id} has no prefix_P.pt"
        )

    print(f"  artifact={artifact.name} -> {prefix_path}")
    return prefix_path, artifact.name


def load_model_and_prefix(run, expected_L, expected_m):
    prefix_path, artifact_name = download_run_prefix(run, expected_L)

    if expected_L == 0:
        model = GPT.from_pretrained(MODEL_TYPE, dict(dropout=DROPOUT))
        model.eval().to(DEVICE)
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model, None, {
            "model_type": MODEL_TYPE,
            "block_size": model.config.block_size,
            "prefix_len": 0,
            "prefix_update_period": expected_m,
            "artifact_name": None,
        }

    saved = torch.load(prefix_path, map_location="cpu")
    saved_model_type = saved.get("model", MODEL_TYPE)
    saved_prefix_len = int(saved.get("prefix_len", expected_L))
    saved_m = int(saved.get("prefix_update_period", expected_m))
    saved_block_size = int(saved.get("block_size", MAX_POSITIONS))

    if saved_prefix_len != expected_L:
        raise ValueError(
            f"Run {run.id} says L={expected_L}, but checkpoint says "
            f"L={saved_prefix_len}"
        )
    if saved_m != expected_m:
        print(
            f"  WARNING: run name says m={expected_m}, checkpoint says m={saved_m}"
        )

    model = GPT.from_pretrained(saved_model_type, dict(dropout=DROPOUT))
    model.eval().to(DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad = False

    P_tensor = saved.get("P")
    if P_tensor is None:
        raise ValueError(f"Checkpoint from run {run.id} has no P tensor")
    expected_shape = (saved_prefix_len, model.config.n_embd)
    if tuple(P_tensor.shape) != expected_shape:
        raise ValueError(
            f"Prefix shape {tuple(P_tensor.shape)} != expected {expected_shape}"
        )

    soft_prefix = SoftPrefix(saved_prefix_len, model.config.n_embd, DEVICE)
    with torch.no_grad():
        soft_prefix.P.copy_(P_tensor.to(DEVICE))
    soft_prefix.eval()

    return model, soft_prefix, {
        "model_type": saved_model_type,
        "block_size": min(saved_block_size, model.config.block_size),
        "prefix_len": saved_prefix_len,
        "prefix_update_period": saved_m,
        "artifact_name": artifact_name,
    }


def get_training_metrics(run):
    keys = [
        "efficiency/tokens_per_sec",
        "efficiency/peak_gpu_mem_gb",
        "efficiency/wall_clock_sec",
        "prefix/cache_hit_ratio",
        "target/hit_ppl_50",
        "target/steps_logged",
        "mfu",
    ]

    values = {}
    try:
        for row in run.scan_history(keys=keys):
            for key in keys:
                value = row.get(key)
                if value is not None:
                    try:
                        if not math.isnan(float(value)):
                            values[key] = value
                    except (TypeError, ValueError):
                        values[key] = value
    except Exception as exc:
        print(f"  WARNING: could not scan metrics for {run.id}: {exc}")

    return {
        "tokens_per_sec": round(
            float(values.get("efficiency/tokens_per_sec", 0)), 1
        ),
        "peak_mem_gb": round(
            float(values.get("efficiency/peak_gpu_mem_gb", 0)), 2
        ),
        "wall_clock_sec": round(
            float(values.get("efficiency/wall_clock_sec", 0)), 1
        ),
        "cache_hit_ratio": round(
            float(values.get("prefix/cache_hit_ratio", 0)), 3
        ),
        "hit_ppl_50": int(values.get("target/hit_ppl_50", 0)),
        "steps_to_target": int(values.get("target/steps_logged", 0)),
        "mfu": round(float(values.get("mfu", 0)), 2),
    }


def estimate_val_perplexity(
    model,
    soft_prefix,
    token_block_size,
    batch_size,
    eval_iters,
    use_cache,
):
    block_size = min(token_block_size, model.config.block_size)
    data = np.memmap(VAL_DATA, dtype=np.uint16, mode="r")
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    losses = []

    prefix_kv = None
    if use_cache and soft_prefix is not None:
        prefix_kv = model.build_prefix_kv(
            soft_prefix,
            batch_size=batch_size,
        )

    with torch.no_grad():
        for _ in range(eval_iters):
            indices = torch.randint(len(data) - block_size, (batch_size,))
            X = torch.stack([
                torch.from_numpy(data[i:i + block_size].astype(np.int64))
                for i in indices
            ]).to(DEVICE)
            Y = torch.stack([
                torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64))
                for i in indices
            ]).to(DEVICE)

            with ctx:
                if prefix_kv is not None:
                    _, loss = model(X, Y, prefix=None, prefix_kv=prefix_kv)
                else:
                    _, loss = model(X, Y, prefix=soft_prefix)
            losses.append(loss.item())

    val_loss = float(np.mean(losses))
    return val_loss, math.exp(val_loss)


results = []
errors = []

for run_id in RUN_IDS:
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        L, m, train_cache = parse_run_identity(run)
        eval_cache = train_cache if EVAL_CACHE_MODE == "match" else bool(
            EVAL_CACHE_MODE
        )

        print(
            f"\nRun {run_id}: {run.name}\n"
            f"  L={L}, m={m}, train_cache={train_cache}, "
            f"eval_cache={eval_cache}"
        )

        model, soft_prefix, metadata = load_model_and_prefix(run, L, m)

        # Reuse identical sampled validation windows for matched configurations.
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        eval_start = time.time()
        val_loss, val_ppl = estimate_val_perplexity(
            model=model,
            soft_prefix=soft_prefix,
            token_block_size=metadata["block_size"],
            batch_size=EVAL_BATCH_SIZE,
            eval_iters=EVAL_ITERS,
            use_cache=eval_cache,
        )
        eval_time = time.time() - eval_start

        result = {
            "run_id": run.id,
            "run_name": run.name,
            "created_at": str(run.created_at),
            "L": metadata["prefix_len"],
            "m": metadata["prefix_update_period"],
            "train_cache": train_cache,
            "eval_cache": eval_cache,
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "param_count": metadata["prefix_len"] * model.config.n_embd,
            "model_type": metadata["model_type"],
            "token_block_size": metadata["block_size"],
            "eval_time_sec": round(eval_time, 2),
            "artifact_name": metadata["artifact_name"],
            "task": TASK,
            "method": "positional",
        }
        result.update(get_training_metrics(run))
        results.append(result)

        print(
            f"  val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}, "
            f"eval_time={eval_time:.1f}s"
        )

        del model, soft_prefix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as exc:
        print(f"  FAILED run {run_id}: {exc}")
        errors.append({"run_id": run_id, "error": str(exc)})


results.sort(key=lambda row: (row["L"], row["m"], row["train_cache"]))

print("\n" + "=" * 104)
print("H2 POSITIONAL EXACT-RUN EVAL SUMMARY")
print("=" * 104)
print(
    f"{'run_id':>8}  {'L':>5}  {'m':>3}  {'train':>5}  {'eval':>5}  "
    f"{'loss':>8}  {'ppl':>8}  {'tok/s':>9}  {'mem':>6}  {'wall(s)':>9}"
)
print("-" * 104)
for row in results:
    print(
        f"{row['run_id']:>8}  {row['L']:>5}  {row['m']:>3}  "
        f"{str(row['train_cache']):>5}  {str(row['eval_cache']):>5}  "
        f"{row['val_loss']:>8.4f}  {row['val_ppl']:>8.2f}  "
        f"{row['tokens_per_sec']:>9.1f}  {row['peak_mem_gb']:>6.2f}  "
        f"{row['wall_clock_sec']:>9.1f}"
    )


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
eval_run = wandb.init(
    project=PROJECT,
    name=f"h2-positional-eval-by-id-cacheonoff-{timestamp}",
    job_type="eval",
)
wandb.log({
    "h2/by_id_summary": wandb.Table(
        columns=[
            "run_id", "run_name", "created_at", "L", "m",
            "train_cache", "eval_cache", "val_loss", "val_ppl",
            "param_count", "model_type", "token_block_size",
            "eval_time_sec", "tokens_per_sec", "peak_mem_gb",
            "wall_clock_sec", "cache_hit_ratio", "mfu",
            "artifact_name",
        ],
        data=[[
            row["run_id"], row["run_name"], row["created_at"],
            row["L"], row["m"], row["train_cache"], row["eval_cache"],
            row["val_loss"], row["val_ppl"], row["param_count"],
            row["model_type"], row["token_block_size"],
            row["eval_time_sec"], row["tokens_per_sec"],
            row["peak_mem_gb"], row["wall_clock_sec"],
            row["cache_hit_ratio"], row["mfu"], row["artifact_name"],
        ] for row in results],
    )
})
wandb.finish()


summary_path = "/kaggle/working/h2_positional_eval_sweep_by_id_cacheonoff.json"
with open(summary_path, "w") as file:
    json.dump({"results": results, "errors": errors}, file, indent=2)

print(f"\nSaved: {summary_path}")
if errors:
    print(f"Runs with errors: {len(errors)}")

