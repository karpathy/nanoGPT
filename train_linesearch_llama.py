"""
Train a LLaMA model with line-search learning rate adaptation for stage2 runs.
"""

import os
import time
import math
import json
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, init_process_group

from checkpoint_utils import resolve_resume_checkpoint
from lr_sched import LineSearchScheduler
from llama.train_support import (
    default_tokenized_cache_dir,
    build_optimizer,
    build_llama_model,
    build_tokenizer,
    load_token_data,
    prepare_token_bin_cache,
    strip_unwanted_prefix,
    token_bin_cache_is_ready,
    wait_for_token_bin_cache,
)

# -----------------------------------------------------------------------------
# default config values
# I/O
out_dir = "/scratch.global/chen8596/out"
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"
# data
dataset = "c4"
dataset_config_name = "en"
gradient_accumulation_steps = 4 * 4
batch_size = 16
block_size = 1024
max_length = 1024
dataloader_num_workers = 4
tokenizer_name = "t5-base"
tokenized_data_cache_dir = ""
hf_data_cache_dir = ""
preprocessing_num_workers = 8
tokenizer_batch_size = 1000
overwrite_tokenized_cache = False
data_backend = "memmap"
# model
llama_config_path = "llama_config/llama_60m.json"
# adamw optimizer
learning_rate = 1e-6
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0
# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 6e-5
# line search settings
linesearch_interval = 0
linesearch_accum_steps = 32
linesearch_num_search = 30
linesearch_start_lr = 0.0
linesearch_c1 = 0.1
linesearch_search_mode = "bisection"
linesearch_factor = 0.5
# DDP settings
backend = "nccl"
# system
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = True
seed = 1337
# experiment controls
experiment_name = ""
trial_id = ""
experiment_metric_mode = "min"
experiment_train_target_value = -1.0
experiment_train_target_enabled = False
experiment_test_target_value = -1.0
experiment_test_target_enabled = False
max_running_time_hours = 0.0
save_last_checkpoint = True
experiment_summary_path = ""
experiment_records_path = ""
prune_signal_path = ""
stop_at_eval_boundary = False
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}
always_save_checkpoint = False
save_last_checkpoint = False
config["always_save_checkpoint"] = False
config["save_last_checkpoint"] = False
linesearch_interval = max(1, int(max_iters * 0.1))
config["linesearch_interval"] = linesearch_interval
# -----------------------------------------------------------------------------

if max_length != block_size:
    raise ValueError("max_length and block_size must match for LLaMA experiments.")

ddp = int(os.environ.get("RANK", -1)) != -1
ddp_rank = int(os.environ.get("RANK", 0))
ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
master_process = ddp_rank == 0
seed_offset = ddp_rank

tokenizer = build_tokenizer(tokenizer_name=tokenizer_name, max_length=max_length)
print(f"tokenizer vocab size: {tokenizer.vocab_size}")
tokenized_data_cache_dir = tokenized_data_cache_dir or default_tokenized_cache_dir(
    dataset_name=dataset,
    dataset_config_name=dataset_config_name,
    tokenizer_name=tokenizer_name,
    max_length=max_length,
)
if master_process:
    print(f"tokenized cache dir: {tokenized_data_cache_dir}")
    prepare_token_bin_cache(
        dataset_name=dataset,
        dataset_config_name=dataset_config_name,
        tokenizer=tokenizer,
        cache_dir=tokenized_data_cache_dir,
        overwrite_tokenized_cache=overwrite_tokenized_cache,
        text_batch_size=tokenizer_batch_size,
        num_proc=preprocessing_num_workers,
    )
elif not token_bin_cache_is_ready(tokenized_data_cache_dir):
    print(f"waiting for token cache to appear: {tokenized_data_cache_dir}")
    wait_for_token_bin_cache(tokenized_data_cache_dir)

if ddp:
    init_process_group(backend=backend)
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    device = device

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

train_data, token_meta = load_token_data(tokenized_data_cache_dir, split="train", data_backend=data_backend)
val_data, _ = load_token_data(tokenized_data_cache_dir, split="val", data_backend=data_backend)
if master_process:
    print(
        f"loaded token cache backend={data_backend}, "
        f"train_tokens={token_meta['train_tokens']:,}, val_tokens={token_meta['val_tokens']:,}"
    )


def get_batch(split, to_device=True):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,)).numpy()
    offsets = np.arange(block_size)
    x = torch.from_numpy(np.asarray(data[ix[:, None] + offsets], dtype=np.int64))
    if not to_device:
        return x, x
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x, x


iter_num = 0
best_train_loss = 1e9
best_val_loss = 1e9

model_args = {
    "llama_config_path": llama_config_path,
    "tokenizer_name": tokenizer_name,
    "vocab_size": tokenizer.vocab_size,
    "max_length": max_length,
    "block_size": block_size,
    "tokenized_data_cache_dir": tokenized_data_cache_dir,
    "data_backend": data_backend,
}

checkpoint = None
if init_from == "scratch":
    print("Initializing a new LLaMA model from scratch")
    model, model_config_dict = build_llama_model(
        llama_config_path=llama_config_path,
        tokenizer_vocab_size=tokenizer.vocab_size,
        max_length=max_length,
    )
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = resolve_resume_checkpoint(out_dir)
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    model_args.update(checkpoint_model_args)
    model, model_config_dict = build_llama_model(
        llama_config_path=model_args["llama_config_path"],
        tokenizer_vocab_size=int(model_args["vocab_size"]),
        max_length=int(model_args["max_length"]),
    )
    state_dict = strip_unwanted_prefix(checkpoint["model"])
    model.load_state_dict(state_dict)
    iter_num = int(checkpoint["iter_num"])
    best_train_loss = float(checkpoint.get("best_train_loss", best_train_loss))
    best_val_loss = float(checkpoint["best_val_loss"])
else:
    raise ValueError(f"Unsupported init_from={init_from}")

print(f"model vocab size: {model.config.vocab_size}")
model.to(device)
if master_process and device_type == "cuda":
    flash_enabled = getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)()
    mem_efficient_enabled = getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: False)()
    math_enabled = getattr(torch.backends.cuda, "math_sdp_enabled", lambda: False)()
    print(
        "sdpa backends: "
        f"flash={flash_enabled}, mem_efficient={mem_efficient_enabled}, math={math_enabled}"
    )

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = build_optimizer(
    model=model,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    betas=(beta1, beta2),
    device_type=device_type,
    optimizer_type="AdamW",
)
scheduler = LineSearchScheduler(
    optimizer=optimizer,
    model_paras=model.parameters(),
    num_search=linesearch_num_search,
    start_lr=linesearch_start_lr,
    optimizer_type="AdamW",
    injection=False,
    search_mode=linesearch_search_mode,
    warmup_length=warmup_iters,
)
if init_from == "resume" and checkpoint is not None:
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    eval_model = model.module if ddp else model
    eval_model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, labels = get_batch(split)
            with ctx:
                outputs = eval_model(input_ids=X, attention_mask=None, labels=labels)
            losses[k] = outputs.loss.item()
        out[split] = losses.mean()
    eval_model.train()
    return out


def save_checkpoint(path):
    checkpoint_payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_args": model_args,
        "model_config": model_config_dict,
        "iter_num": iter_num,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    print(f"saving checkpoint to {path}")
    torch.save(checkpoint_payload, path)


def write_experiment_summary(termination_reason, elapsed_hours):
    if not experiment_summary_path or not master_process:
        return
    summary = {
        "experiment_name": experiment_name,
        "trial_id": trial_id,
        "train_script": "train_linesearch_llama.py",
        "out_dir": out_dir,
        "best_checkpoint_path": os.path.join(out_dir, "ckpt.pt") if os.path.exists(os.path.join(out_dir, "ckpt.pt")) else "",
        "last_checkpoint_path": os.path.join(out_dir, "ckpt_last.pt") if os.path.exists(os.path.join(out_dir, "ckpt_last.pt")) else "",
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "iter_num": int(iter_num),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "metric_mode": experiment_metric_mode,
        "wall_clock_hours": float(elapsed_hours),
        "forward_backward_hours": float(elapsed_hours),
        "forward_hours": float(forward_seconds / 3600.0),
        "backward_hours": float(backward_seconds / 3600.0),
        "elapsed_wall_clock_hours": float((time.time() - train_start_time) / 3600.0),
        "termination_reason": termination_reason,
    }
    with open(experiment_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def append_experiment_record(step, train_loss, val_loss, wall_clock_hours):
    if not experiment_records_path or not master_process:
        return
    record = {
        "stage": "stage2",
        "trial_id": trial_id,
        "step": int(step),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "wall_clock_hours": float(wall_clock_hours),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    with open(experiment_records_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def prune_requested():
    return bool(prune_signal_path) and os.path.exists(prune_signal_path)


def training_clock_now():
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    return time.time()


def should_stop_at_eval_boundary():
    if not stop_at_eval_boundary:
        return False
    return (iter_num + eval_interval) > max_iters


X, labels = get_batch("train")
train_start_time = time.time()
forward_backward_seconds = 0.0
forward_seconds = 0.0
backward_seconds = 0.0
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
termination_reason = "max_iters_reached"
while True:
    if iter_num % linesearch_interval == 0 or (warmup_iters > 0 and iter_num <= warmup_iters and iter_num % warmup_iters == 0):
        print("LINESEARCH!!!")
        fixed_batches = []
        for _ in range(linesearch_accum_steps):
            x_ls, y_ls = get_batch("train", to_device=False)
            fixed_batches.append((x_ls, y_ls))

        def make_closure():
            def line_search_closure(require_grad=False):
                device_for_batch = next(model.parameters()).device
                total_loss = torch.zeros((), device=device_for_batch)
                sync_last_micro_step = len(fixed_batches) - 1
                for micro_step, (x_ls, labels_ls) in enumerate(fixed_batches):
                    if device_type == "cuda":
                        x_ls = x_ls.pin_memory().to(device_for_batch, non_blocking=True)
                    else:
                        x_ls = x_ls.to(device_for_batch)
                    labels_ls = x_ls
                    if ddp and require_grad:
                        model.require_backward_grad_sync = (micro_step == sync_last_micro_step)
                    with ctx:
                        outputs_ls = model(
                            input_ids=x_ls,
                            attention_mask=None,
                            labels=labels_ls,
                        )
                    total_loss += outputs_ls.loss.detach()

                    if require_grad:
                        (outputs_ls.loss / (linesearch_accum_steps)).backward()
                        if grad_clip != 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if ddp and require_grad:
                    model.require_backward_grad_sync = True
                avg_loss = total_loss / (linesearch_accum_steps)
                return avg_loss.item()

            return line_search_closure

        line_search_closure = make_closure()

    c1_use = linesearch_c1 + (1 - linesearch_c1) * (iter_num / max_iters)
    scheduler.step(
        line_search_closure,
        c1=c1_use,
        factor=linesearch_factor,
        step=iter_num,
        interval=linesearch_interval,
        condition="armijo",
        warmup_length=warmup_iters,
    )

    if iter_num % eval_interval == 0:
        should_terminate = False
        if master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            best_train_loss = min(best_train_loss, losses["train"].item())
            append_experiment_record(
                step=iter_num,
                train_loss=losses["train"].item(),
                val_loss=losses["val"].item(),
                wall_clock_hours=forward_backward_seconds / 3600.0,
            )
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    save_checkpoint(os.path.join(out_dir, "ckpt.pt"))
            if prune_requested():
                termination_reason = "pruned"
                should_terminate = True
            elif should_stop_at_eval_boundary():
                termination_reason = "eval_budget_boundary_reached"
                should_terminate = True
        if ddp:
            termination_flag = torch.tensor([int(should_terminate)], device=device)
            dist.broadcast(termination_flag, src=0)
            should_terminate = bool(termination_flag.item())
        if should_terminate:
            break
    if iter_num == 0 and eval_only:
        termination_reason = "eval_only"
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        forward_start_time = training_clock_now()
        with ctx:
            outputs = model(input_ids=X, attention_mask=None, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
        forward_seconds += training_clock_now() - forward_start_time
        X, labels = get_batch("train")
        backward_start_time = training_clock_now()
        scaler.scale(loss).backward()
        backward_seconds += training_clock_now() - backward_start_time
    forward_backward_seconds = forward_seconds + backward_seconds

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5 and hasattr(raw_model, "estimate_mfu"):
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        termination_reason = "max_iters_reached"
        break
    if max_running_time_hours > 0:
        training_hours = forward_backward_seconds / 3600.0
        if training_hours >= max_running_time_hours:
            termination_reason = "time_limit_reached"
            break

if master_process and save_last_checkpoint and iter_num > 0:
    save_checkpoint(os.path.join(out_dir, "ckpt_last.pt"))
write_experiment_summary(
    termination_reason=termination_reason,
    elapsed_hours=forward_backward_seconds / 3600.0,
)
if ddp:
    destroy_process_group()
