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
import json
from contextlib import nullcontext
from lr_sched import LineSearchScheduler
import logging

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from checkpoint_utils import resolve_resume_checkpoint
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = '/scratch.global/chen8596/out'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 4 * 4 # used to simulate larger batch sizes
batch_size = 30 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
data_backend = 'ram' # 'memmap' keeps files mmaped, 'ram' loads them into process memory
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 1e-6 # max learning rate
max_iters = 5000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100  # how many steps to warm up for
lr_decay_iters = 5000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# line search settings
linesearch_interval = 0
linesearch_accum_steps = 32
linesearch_num_search = 30
linesearch_start_lr = 0.0
linesearch_c1 = 0.1
linesearch_search_mode = 'bisection'
linesearch_factor = 0.5
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
print("DTYPE", dtype)
compile = True # use PyTorch 2.0 to compile the model to be faster
use_flash_attention = True # use PyTorch scaled_dot_product_attention when available
# experiment controls
experiment_name = ''
trial_id = ''
experiment_metric_mode = 'min'
experiment_train_target_value = -1.0
experiment_train_target_enabled = False
experiment_test_target_value = -1.0
experiment_test_target_enabled = False
max_running_time_hours = 0.0
save_last_checkpoint = True
experiment_summary_path = ''
experiment_records_path = ''
prune_signal_path = ''
stop_at_eval_boundary = False
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
always_save_checkpoint = False
save_last_checkpoint = False
config['always_save_checkpoint'] = False
config['save_last_checkpoint'] = False
linesearch_interval = max(1, int(max_iters * 0.1))
config['linesearch_interval'] = linesearch_interval
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
data_dir = os.path.join('/scratch.global/chen8596/nanogpt_data', dataset)
def load_token_data(filename):
    path = os.path.join(data_dir, filename)
    if data_backend == 'memmap':
        return np.memmap(path, dtype=np.uint16, mode='r')
    if data_backend == 'ram':
        if master_process:
            print(f"loading {path} into RAM")
        return np.fromfile(path, dtype=np.uint16)
    raise ValueError(f"Unsupported data_backend: {data_backend}")

train_data = load_token_data('train.bin')
val_data = load_token_data('val.bin')
if master_process:
    print(f"using data backend: {data_backend}")

def get_batch_linesearch(split, to_device=True):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,)).numpy()
    offsets = np.arange(block_size)
    x = torch.from_numpy(data[ix[:, None] + offsets]).long()
    y = torch.from_numpy(data[ix[:, None] + offsets + 1]).long()
    if not to_device:
        return x, y
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_train_loss = 1e9
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
                  bias=bias, vocab_size=None, dropout=dropout,
                  use_flash_attention=use_flash_attention) # start with model_args from command line
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
    ckpt_path = resolve_resume_checkpoint(out_dir)
    print(f"Loading checkpoint from {ckpt_path}")
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
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)
if master_process:
    print(f"using flash attention: {model.transformer.h[0].attn.flash}")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, 1e-6, (beta1, beta2), device_type)
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
            X, Y = get_batch_linesearch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# define the linesearch lr_sched
scheduler = LineSearchScheduler(optimizer=optimizer, 
                                model_paras=model.parameters(), 
                                num_search=linesearch_num_search, start_lr=linesearch_start_lr, 
                                optimizer_type="AdamW", 
                                injection=False, 
                                search_mode=linesearch_search_mode, 
                                warmup_length=warmup_iters)

def save_checkpoint(path):
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    print(f"saving checkpoint to {path}")
    torch.save(checkpoint, path)

def write_experiment_summary(
    termination_reason,
    forward_backward_hours,
    wall_clock_hours,
):
    if not experiment_summary_path or not master_process:
        return
    summary = {
        'experiment_name': experiment_name,
        'trial_id': trial_id,
        'train_script': 'train_linesearch.py',
        'out_dir': out_dir,
        'best_checkpoint_path': os.path.join(out_dir, 'ckpt.pt') if os.path.exists(os.path.join(out_dir, 'ckpt.pt')) else '',
        'last_checkpoint_path': os.path.join(out_dir, 'ckpt_last.pt') if os.path.exists(os.path.join(out_dir, 'ckpt_last.pt')) else '',
        'best_train_loss': float(best_train_loss),
        'best_val_loss': float(best_val_loss),
        'iter_num': int(iter_num),
        'learning_rate': float(optimizer.param_groups[0]['lr']),
        'metric_mode': experiment_metric_mode,
        'wall_clock_hours': float(wall_clock_hours),
        'forward_backward_hours': float(forward_backward_hours),
        'forward_hours': float(forward_seconds / 3600.0),
        'backward_hours': float(backward_seconds / 3600.0),
        'elapsed_wall_clock_hours': float(wall_clock_hours),
        'termination_reason': termination_reason,
    }
    with open(experiment_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def append_experiment_record(step, train_loss, val_loss, forward_backward_hours, wall_clock_hours):
    if not experiment_records_path or not master_process:
        return
    record = {
        'stage': 'stage2',
        'trial_id': trial_id,
        'step': int(step),
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'wall_clock_hours': float(wall_clock_hours),
        'forward_backward_hours': float(forward_backward_hours),
        'learning_rate': float(optimizer.param_groups[0]['lr']),
    }
    with open(experiment_records_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, sort_keys=True) + '\n')

def prune_requested():
    return bool(prune_signal_path) and os.path.exists(prune_signal_path)

def training_clock_now():
    if device_type == 'cuda':
        torch.cuda.synchronize(device)
    return time.time()


def should_stop_at_eval_boundary():
    if not stop_at_eval_boundary:
        return False
    return (iter_num + eval_interval) > max_iters

# learning rate decay scheduler (cosine with warmup)


# training loop
X, Y = get_batch_linesearch('train') # fetch the very first batch

train_start_time = time.time()
forward_backward_seconds = 0.0
forward_seconds = 0.0
backward_seconds = 0.0
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
termination_reason = 'max_iters_reached'
while True:
    # print(iter_num, linesearch_interval, warmup_iters)
    if iter_num % linesearch_interval == 0 or (iter_num <= warmup_iters and iter_num % warmup_iters == 0):
        print("LINESEARCH!!!")
        fixed_batches = []
        for i in range(linesearch_accum_steps):
            X_ls, Y_ls = get_batch_linesearch("train", to_device=False)
            fixed_batches.append((X_ls, Y_ls))

        def make_closure():
            def line_search_closure(require_grad=False):
                device = next(model.parameters()).device
                total_loss = torch.zeros((), device=device)
                sync_last_micro_step = len(fixed_batches) - 1
                for micro_step, (x_ls, y_ls) in enumerate(fixed_batches):
                    if device_type == 'cuda':
                        x_ls = x_ls.pin_memory().to(device, non_blocking=True)
                        y_ls = y_ls.pin_memory().to(device, non_blocking=True)
                    else:
                        x_ls = x_ls.to(device)
                        y_ls = y_ls.to(device)
                    if ddp and require_grad:
                        # Match the main training loop and only synchronize DDP
                        # gradients on the last accumulation step.
                        model.require_backward_grad_sync = (micro_step == sync_last_micro_step)
                    with ctx:
                        _, loss_ls = model(x_ls, y_ls)

                    # Keep only the scalar value so we do not retain every
                    # micro-batch autograd graph until the closure returns.
                    total_loss += loss_ls.detach()

                    if require_grad:
                        (loss_ls / (linesearch_accum_steps)).backward()

                        # Clip gradients produced by the closure to keep them consistent
                        # with the main training loop. This prevents excessively large
                        # gradient norms from affecting line-search behavior.
                        if grad_clip != 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if ddp and require_grad:
                    model.require_backward_grad_sync = True

                avg_loss = total_loss / (linesearch_accum_steps + 1)
                return avg_loss.item()
            return line_search_closure
        line_search_closure = make_closure()
            
    c1_use = linesearch_c1 + (1 - linesearch_c1) * (iter_num / max_iters)
    ls_start_time = training_clock_now()
    scheduler.step(
        line_search_closure,
        c1=c1_use,
        factor=linesearch_factor,
        step=iter_num,
        interval=linesearch_interval,
        condition="armijo",
        warmup_length=warmup_iters,
    )
    forward_backward_seconds += training_clock_now() - ls_start_time
    lr = optimizer.param_groups[0]['lr']


    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        should_terminate = False
        if master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            best_train_loss = min(best_train_loss, losses['train'].item())
            append_experiment_record(
                step=iter_num,
                train_loss=losses['train'].item(),
                val_loss=losses['val'].item(),
                forward_backward_hours=forward_backward_seconds / 3600.0,
                wall_clock_hours=(time.time() - train_start_time) / 3600.0,
            )
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    save_checkpoint(os.path.join(out_dir, 'ckpt.pt'))
            if prune_requested():
                termination_reason = 'pruned'
                should_terminate = True
            elif should_stop_at_eval_boundary():
                termination_reason = 'eval_budget_boundary_reached'
                should_terminate = True
        if ddp:
            termination_flag = torch.tensor([int(should_terminate)], device=device)
            dist.broadcast(termination_flag, src=0)
            should_terminate = bool(termination_flag.item())
        if should_terminate:
            break
    if iter_num == 0 and eval_only:
        termination_reason = 'eval_only'
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
        forward_start_time = training_clock_now()
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        forward_seconds += training_clock_now() - forward_start_time
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch_linesearch('train')
        # backward pass, with gradient scaling if training in fp16
        backward_start_time = training_clock_now()
        scaler.scale(loss).backward()
        backward_seconds += training_clock_now() - backward_start_time
    forward_backward_seconds = forward_seconds + backward_seconds
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
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
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        termination_reason = 'max_iters_reached'
        break
    if max_running_time_hours > 0:
        training_hours = forward_backward_seconds / 3600.0
        if training_hours >= max_running_time_hours:
            termination_reason = 'time_limit_reached'
            break

if master_process and save_last_checkpoint and iter_num > 0:
    save_checkpoint(os.path.join(out_dir, 'ckpt_last.pt'))
write_experiment_summary(
    termination_reason=termination_reason,
    forward_backward_hours=forward_backward_seconds / 3600.0,
    wall_clock_hours=(time.time() - train_start_time) / 3600.0,
)
if ddp:
    destroy_process_group()
