"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, SoftPrefix, DeepPrefix

import re
import tiktoken

enc = tiktoken.get_encoding("gpt2")
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
prefix_type = 'soft'
prefix_len = 0              # L: number of soft tokens (0 = no prefix)
prefix_update_period = 1    # m: update P every m steps (1 = dense)
prefix_cache = True         # whether to cache prefix KV between updates

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

# this is used for GSM8K dataset, supervised
supervised = os.path.exists(os.path.join(data_dir, 'train.pt'))

if supervised:
    train_data = torch.load(os.path.join(data_dir, 'train.pt'))
    val_data = torch.load(os.path.join(data_dir, 'val.pt'))
    test_data = torch.load(os.path.join(data_dir, 'test.pt'))

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data), (batch_size,))

        x_batch, y_batch = [], []
        for i in ix:
            ex = data[i]
            ids = ex['input_ids'][:block_size + 1]
            lab = ex['labels'][:block_size + 1]
            pad_len = (block_size + 1) - len(ids)
            ids = ids + [0] * pad_len
            lab = lab + [-1] * pad_len
            x_batch.append(torch.tensor(ids[:-1], dtype=torch.long))
            y_batch.append(torch.tensor(lab[1:], dtype=torch.long))

        x = torch.stack(x_batch)
        y = torch.stack(y_batch)
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

else:
    def get_batch(split):
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

prefix_module = None
is_prefix_tuning = False
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    checkpoint_prefix = checkpoint.get('soft_prefix')
    if checkpoint_prefix is not None:
        prefix_len = checkpoint.get('prefix_len', prefix_len)
        prefix_type = checkpoint.get('prefix_type', prefix_type)
        prefix_update_period = checkpoint.get('prefix_update_period', prefix_update_period)
        for p in model.parameters():
            p.requires_grad = False
        if prefix_type == 'soft':
            with torch.no_grad():
                indices = torch.randint(0, model.config.vocab_size, (prefix_len,))
                init_embeddings = model.transformer.wte.weight[indices].clone()
            prefix_module = SoftPrefix(prefix_len, model.config.n_embd, device,
                                       init_embeddings=init_embeddings).to(device)
        elif prefix_type == 'deep':
            ...
        is_prefix_tuning = True
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    if prefix_len > 0:
        for p in model.parameters():
            p.requires_grad = False
        if prefix_type == 'soft':
            prefix_module = SoftPrefix(prefix_len, model.config.n_embd, device).to(device)
        elif prefix_type == 'deep':
            prefix_module = DeepPrefix(prefix_len, model.config.n_embd, model.config.n_layer, device).to(device)
        is_prefix_tuning = True
        print(f"{prefix_type} prefix: L={prefix_len}")
    else:
        print("Finetuning all GPT-2 parameters (prefix_len=0)")
        print("No soft prefix — frozen GPT-2 baseline (L=0)")

model.to(device)
if is_prefix_tuning:
    effective_block_size = model.config.block_size - prefix_len
    assert effective_block_size > 0, "prefix_len must be smaller than the model block size"
    if block_size > effective_block_size:
        print(f"Reducing training block_size from {block_size} to {effective_block_size} to fit prefix")
        block_size = effective_block_size
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
if is_prefix_tuning:
    prefix_params = [prefix_module.P] if isinstance(prefix_module, SoftPrefix) else prefix_module.parameters()
    optimizer = torch.optim.AdamW(prefix_params, lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
else:
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y, prefix=prefix_module)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

def extract_number(text):
    if "####" in text:
        after = text.split("####")[-1].strip().replace(",", "")
        match = re.search(r'-?\d+\.?\d*', after)
        if match:
            return match.group()
    text_clean = text.replace(",", "")
    numbers = re.findall(r'-?\d+\.?\d*', text_clean)
    if numbers:
        return numbers[-1]
    return None


# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
training_start_time = time.time()
if device_type == 'cuda':
    torch.cuda.reset_peak_memory_stats()

# this is used for gsm8k, in eval mode
@torch.no_grad()
def evaluate_em(data, prefix_module, max_new_tokens=128, use_cache=False):
    raw_model.eval()
    correct = 0
    total = 0

    prefix_kv = None
    if use_cache and prefix_module is not None:
        with torch.no_grad():
            prefix_kv = raw_model.build_prefix_kv(prefix_module, batch_size=1)

    for ex in data:
        prompt_ids = ex['input_ids'][:ex['prompt_len']]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        if prefix_kv is not None:
            out = raw_model.generate(idx, max_new_tokens, temperature=1.0, top_k=1, prefix_kv=prefix_kv)
        else:
            out = raw_model.generate(idx, max_new_tokens, temperature=1.0, top_k=1, prefix=prefix_module)
        generated_text = enc.decode(out[0].tolist())
        pred = extract_number(generated_text)
        gold = extract_number(ex['gold_answer'])
        if pred is not None and gold is not None and pred == gold:
            correct += 1
        total += 1
    raw_model.train()
    return correct / total if total > 0 else 0.0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    update_now = (iter_num % prefix_update_period == 0)
    # on m-update steps, prefix gets gradients
    # on in-between steps, prefix is frozen
    if is_prefix_tuning:
        prefix_module.requires_grad_(update_now)
        if update_now and prefix_cache:
            prefix_module.invalidate()

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            import math

            # wall-clock and throughput ───────────────────────
            wall_clock_elapsed = time.time() - training_start_time
            tokens_seen = iter_num * batch_size * block_size * gradient_accumulation_steps
            tokens_per_sec = tokens_seen / max(wall_clock_elapsed, 1)

            #  early-phase target (adjust threshold to your baseline PPL) ──

            cache_hit_ratio = (
                1.0 - (1.0 / prefix_update_period)
                if prefix_module is not None and prefix_cache and prefix_update_period > 1
                else 0.0
            )

            # common metrics logged for both modes
            common_metrics = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
                "prefix/len": prefix_len,
                "prefix/update_period": prefix_update_period,
                "prefix/param_count": prefix_len * raw_model.config.n_embd if is_prefix_tuning else 0,
                "prefix/cache_hit_ratio": cache_hit_ratio,
                "efficiency/peak_gpu_mem_gb": (
                    torch.cuda.max_memory_allocated() / 1e9 if device_type == 'cuda' else 0.0
                ),
                "prefix/cache_enabled": int(prefix_cache),
                "prefix/update_now": int(update_now) if is_prefix_tuning else 0,
                "efficiency/wall_clock_sec": wall_clock_elapsed,
                "efficiency/tokens_per_sec": tokens_per_sec,
            }

            if not supervised:
                # language modeling mode — log perplexity
                TARGET_PPL = 50.0
                val_ppl = math.exp(losses['val'])
                train_ppl = math.exp(losses['train'])
                hit_target = 1 if val_ppl <= TARGET_PPL else 0

                common_metrics.update({
                    "val/perplexity": val_ppl,
                    "train/perplexity": train_ppl,
                    "target/hit_ppl_50": hit_target,
                    "target/steps_logged": iter_num if hit_target else 0,
                })

            else:
                # supervised mode (gsm8k) — log EM, but not every eval
                # EM requires generation and is slow, so run it less frequently
                if iter_num % (eval_interval * 5) == 0 or iter_num >= max_iters:
                    em_score = evaluate_em(val_data, prefix_module)
                    common_metrics["val/em"] = em_score

            wandb.log(common_metrics)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'soft_prefix': prefix_module.P.detach().cpu() if isinstance(prefix_module, SoftPrefix) else None,
                    'prefix_type': prefix_type if is_prefix_tuning else None,
                    'prefix_len': prefix_len if is_prefix_tuning else 0,
                    'prefix_update_period': prefix_update_period if is_prefix_tuning else 1,
                }
                if not is_prefix_tuning:
                    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                    print(f"saving checkpoint to {ckpt_path}")
                    torch.save(checkpoint, ckpt_path)
                if is_prefix_tuning:
                    prefix_path = os.path.join(out_dir, 'prefix_P.pt')
                    torch.save({
                        'P': prefix_module.P.detach().cpu() if isinstance(prefix_module, SoftPrefix) else None,
                        'prefix_type': prefix_type,
                        'prefix_len': prefix_len,
                        'prefix_update_period': prefix_update_period,
                        'val_loss': float(best_val_loss),
                        'iter_num': iter_num,
                        'task': dataset,
                        'model': init_from,
                        'block_size': block_size,
                        'prefix_cache': prefix_cache,
                        'peak_gpu_mem_gb': (
                            torch.cuda.max_memory_allocated() / 1e9 if device_type == 'cuda' else 0.0
                        ),
                    }, prefix_path)
                    print(f"saving prefix to {prefix_path}")

                    if wandb_log:
                        artifact = wandb.Artifact(
                            name=wandb_run_name.replace(' ', '-'),
                            type="model",
                            metadata={"val_loss": float(best_val_loss), "iter": iter_num}
                        )
                        artifact.add_file(prefix_path)
                        wandb.log_artifact(artifact)
                else:
                    # L=0 baseline — save a small metadata file
                    prefix_path = os.path.join(out_dir, 'prefix_P.pt')
                    torch.save({
                        'P': None,
                        'val_loss': float(best_val_loss),
                        'iter_num': iter_num,
                        'task': dataset,
                        'model': init_from,
                        'block_size': block_size,
                        'prefix_cache': prefix_cache,
                        'peak_gpu_mem_gb': (
                            torch.cuda.max_memory_allocated() / 1e9 if device_type == 'cuda' else 0.0
                        ),
                    }, prefix_path)
                    if wandb_log:
                        artifact = wandb.Artifact(
                            name=wandb_run_name.replace(' ', '-'),
                            type="model",
                            metadata={"val_loss": float(best_val_loss), "iter": iter_num}
                        )
                        artifact.add_file(prefix_path)
                        wandb.log_artifact(artifact)

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if is_prefix_tuning and prefix_cache and not update_now:
                if not prefix_module.cache_valid:
                    with torch.no_grad():
                        prefix_module.cached_kv = raw_model.build_prefix_kv(prefix_module, batch_size=X.size(0))
                        prefix_module.cache_valid = True
                logits, loss = model(X, Y, prefix=None, prefix_kv=prefix_module.cached_kv)
            else:
                logits, loss = model(X, Y, prefix=prefix_module, prefix_kv=None)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        if not is_prefix_tuning or update_now:
            scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0 and (not is_prefix_tuning or update_now):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            prefix_module.parameters() if is_prefix_tuning else model.parameters(),
            grad_clip
        )
    # step the optimizer and scaler if training in fp16
    # flush the gradients as soon as we can, no need for this memory anymore
    if not is_prefix_tuning or update_now:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, update={int(update_now)}")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if master_process and supervised:
    final_em = evaluate_em(test_data, prefix_module)
    print(f"final test EM: {final_em:.4f}")
    if wandb_log:
        wandb.log({"test/em": final_em})


if ddp:
    destroy_process_group()
