"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train_linesearch_muon.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train_linesearch_muon.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train_linesearch_muon.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train_linesearch_muon.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import pickle
import time
import json
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from checkpoint_utils import resolve_resume_checkpoint
from lr_sched_muon_split_armijo import LineSearchScheduler
from model import GPT, GPTConfig
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = '/scratch.global/chen8596/out'
eval_interval = 2000
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
# optimizer
learning_rate = 1.0 # global multiplier for Adam groups only
max_iters = 10000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
muon_lr = 0.0
muon_momentum = 0.95
# adam_head_lr = 0.22
# adam_embed_lr = 0.6
# adam_scalar_lr = 0.04
adam_head_lr = 6e-4
adam_embed_lr = 6e-4
adam_scalar_lr = 6e-4
adam_eps = 1e-10
# learning rate decay settings
decay_lr = True # whether to decay the global lr multiplier for Adam groups
warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 10000 # should be ~= max_iters per Chinchilla
min_lr = 0.1 # minimum lr multiplier
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
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16'
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
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
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
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
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

def get_batch(split, to_device=True):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,)).numpy()
    offsets = np.arange(block_size)
    x = torch.from_numpy(data[ix[:, None] + offsets]).long()
    y = torch.from_numpy(data[ix[:, None] + offsets + 1]).long()
    if not to_device:
        return x, y
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume'
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
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    use_flash_attention=use_flash_attention,
)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    model = GPT(GPTConfig(**model_args))
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = resolve_resume_checkpoint(out_dir)
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"Unsupported init_from={init_from}")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)
if master_process:
    print(f"using flash attention: {model.transformer.h[0].attn.flash}")

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

def build_optimizer(model):
    named_params = list(model.named_parameters())
    head_params = [model.lm_head.weight]
    embed_params = [
        p for n, p in named_params
        if (n.startswith('transformer.wte.') or n.startswith('transformer.wpe.')) and id(p) != id(model.lm_head.weight)
    ]
    scalar_params = [p for _, p in named_params if p.ndim < 2]
    excluded = {id(p) for p in head_params + embed_params + scalar_params}
    hidden_matrix_params = [p for _, p in named_params if p.ndim >= 2 and id(p) not in excluded]

    adam_groups = [
        dict(params=head_params, lr=adam_head_lr, weight_decay=weight_decay),
        dict(params=embed_params, lr=adam_embed_lr, weight_decay=weight_decay),
        dict(params=scalar_params, lr=adam_scalar_lr, weight_decay=weight_decay),
    ]
    adam_groups = [dict(**group, betas=(beta1, beta2), eps=adam_eps, use_muon=False) for group in adam_groups]
    muon_group = dict(
        params=hidden_matrix_params,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=weight_decay,
        use_muon=True,
    )

    optimizer_cls = MuonWithAuxAdam if ddp else SingleDeviceMuonWithAuxAdam
    optimizer = optimizer_cls([*adam_groups, muon_group])
    base_group_lrs = [group['lr'] for group in optimizer.param_groups]
    muon_group_idx = len(optimizer.param_groups) - 1
    return optimizer, base_group_lrs, muon_group_idx, hidden_matrix_params

optimizer, base_group_lrs, muon_group_idx, muon_params = build_optimizer(model)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

scheduler = LineSearchScheduler(
    optimizer=optimizer,
    model_paras=model.parameters(),
    num_search=linesearch_num_search,
    start_lr=linesearch_start_lr,
    optimizer_type="Muon",
    injection=False,
    search_mode=linesearch_search_mode,
    warmup_length=warmup_iters,
    controlled_group_indices=[muon_group_idx],
)
if init_from == 'resume' and checkpoint is not None and 'scheduler' in checkpoint:
    scheduler.load_state_dict(checkpoint['scheduler'])
checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def should_stop_at_eval_boundary():
    if not stop_at_eval_boundary:
        return False
    return (iter_num + eval_interval) > max_iters


def write_experiment_summary(termination_reason, forward_backward_hours, wall_clock_hours, forward_seconds, backward_seconds):
    if not experiment_summary_path or not master_process:
        return
    summary = {
        'experiment_name': experiment_name,
        'trial_id': trial_id,
        'train_script': 'train_linesearch_muon.py',
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
        'muon_lr': float(optimizer.param_groups[muon_group_idx]['lr']),
    }
    
    with open(experiment_records_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, sort_keys=True) + '\n')


def prune_requested():
    return bool(prune_signal_path) and os.path.exists(prune_signal_path)


def training_clock_now():
    if device_type == 'cuda':
        torch.cuda.synchronize(device)
    return time.time()

X, Y = get_batch('train')
train_start_time = time.time()
forward_backward_seconds = 0.0
forward_seconds = 0.0
backward_seconds = 0.0
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
termination_reason = 'max_iters_reached'




while True:
    lr_mult = get_lr(iter_num) if decay_lr else learning_rate
    for group_idx, (param_group, base_lr) in enumerate(zip(optimizer.param_groups, base_group_lrs)):
        if group_idx == muon_group_idx:
            continue
        param_group['lr'] = base_lr * lr_mult

    if iter_num % linesearch_interval == 0 or (warmup_iters > 0 and iter_num <= warmup_iters and iter_num % warmup_iters == 0):
        fixed_batches = [get_batch('train', to_device=False) for _ in range(linesearch_accum_steps)]

        def make_closure():
            def line_search_closure(require_grad=False):
                device_curr = next(model.parameters()).device
                total_loss = torch.zeros((), device=device_curr)
                sync_last_micro_step = len(fixed_batches) - 1
                for micro_step, (x_ls, y_ls) in enumerate(fixed_batches):
                    if device_type == 'cuda':
                        x_ls = x_ls.pin_memory().to(device_curr, non_blocking=True)
                        y_ls = y_ls.pin_memory().to(device_curr, non_blocking=True)
                    else:
                        x_ls = x_ls.to(device_curr)
                        y_ls = y_ls.to(device_curr)
                    if ddp and require_grad:
                        model.require_backward_grad_sync = (micro_step == sync_last_micro_step)
                    with ctx:
                        _, loss_ls = model(x_ls, y_ls)
                    total_loss += loss_ls.detach()
                    if require_grad:
                        (loss_ls / (linesearch_accum_steps)).backward()
                        if grad_clip != 0.0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if ddp and require_grad:
                    model.require_backward_grad_sync = True
                avg_loss = total_loss / (linesearch_accum_steps)
                return avg_loss.item()
            return line_search_closure

        line_search_closure = make_closure()

    c1_use = linesearch_c1 + (1 - linesearch_c1) * (iter_num / max_iters)
    ls_start_time = training_clock_now()
    scheduler.step(
        line_search_closure,
        c1=c1_use,
        step=iter_num,
        interval=linesearch_interval,
        condition="armijo",
        warmup_length=warmup_iters,
        factor=linesearch_factor,
        start_lr=muon_lr,
    )
    forward_backward_seconds += training_clock_now() - ls_start_time

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
            if prune_requested():
                termination_reason = 'pruned'
                should_terminate = True
            if should_stop_at_eval_boundary():
                termination_reason = 'eval_budget_boundary_reached'
                should_terminate = True
        if ddp:
            termination_flag = torch.tensor([int(should_terminate)], device=device)
            torch.distributed.broadcast(termination_flag, src=0)
            should_terminate = bool(termination_flag.item())
        if should_terminate:
            break
    if iter_num == 0 and eval_only:
        termination_reason = 'eval_only'
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        forward_start_time = training_clock_now()
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        forward_seconds += training_clock_now() - forward_start_time
        X, Y = get_batch('train')
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
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
            f"mfu {running_mfu*100:.2f}%, head_lr {optimizer.param_groups[0]['lr']:.4f}, "
            f"muon_lr {optimizer.param_groups[muon_group_idx]['lr']:.4f}"
        )
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        termination_reason = 'max_iters_reached'
        break
    if max_running_time_hours > 0:
        training_hours = forward_backward_seconds / 3600.0
        if training_hours >= max_running_time_hours:
            termination_reason = 'time_limit_reached'
            break

write_experiment_summary(
    termination_reason=termination_reason,
    forward_backward_hours=forward_backward_seconds / 3600.0,
    wall_clock_hours=(time.time() - train_start_time) / 3600.0,
    forward_seconds=forward_seconds,
    backward_seconds=backward_seconds,
)
if ddp:
    destroy_process_group()
