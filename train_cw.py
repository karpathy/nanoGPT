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

from model import GPTConfig, GPT

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
use_codebook = False # True: feed each token's DAC codebook vector (data/<dataset>/codebook.npy) instead of a learned embedding
eeg_dataloader = False # True: windows from data/<dataset>/{train,val}/*.npz, never across channels, recordings or 60 s chunks
pos_encoding = 'learned' # 'learned' (wpe, original) | 'sinusoidal' (fixed sin/cos table)
bidirectional = False # True: masked-token prediction with full attention (BERT) instead of next-token (GPT)
mask_ratio = 0.15 # bidirectional only: fraction of tokens replaced by MASK and predicted
multichannel = False # True: windows of [channels, frames] from the per-recording grids (needs eeg_dataloader and bidirectional)
max_channels = 94 # multichannel: rows per window (fewer are padded); block_size must be >= max_channels * window_frames
window_frames = 32 # multichannel: frames per channel per window (8 per second -> 4 s)
channel_select = 'standard94' # 'standard94' (one channel per standard location, within 2 cm) | 'all'
channel_encoding = 'none' # multichannel: 'none' (control) | 'learned' (one learned vector per standard location)

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
# data_dir = os.path.join('data', dataset)
data_dir = dataset # (CW) - was above ^
if eeg_dataloader:
    from eeg_dataloader import EEGDataLoader
    np.random.seed(1337 + seed_offset)   # the loader samples with numpy; torch.manual_seed above does not cover it
    if multichannel:
        assert bidirectional and max_channels * window_frames <= block_size, "multichannel needs bidirectional and block_size >= max_channels * window_frames"
    loaders = {split: EEGDataLoader(data_dir, split, window_frames if multichannel else block_size) for split in ('train', 'val')}
def get_batch(split):
    kw = {}                                                    # extra model inputs (multichannel only)
    if eeg_dataloader and multichannel:
        x, kw = loaders[split].get_grid_batch(batch_size, max_channels, channel_select)
        y = x                                                  # replaced by the masking below
    elif eeg_dataloader:
        x, y = loaders[split].get_batch(batch_size)
    else:
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if bidirectional: #jm
        mask = torch.rand(x.shape) < mask_ratio            # which positions to hide
        if multichannel:
            mask &= kw['valid'][:, :, None]                    # only real channels; padding keeps target -1
        y = torch.where(mask, x, torch.full_like(x, -1))   # target: the original token where masked, -1 (ignored) elsewhere
        x = x.masked_fill(mask, mask_token_id)             # input: MASK id at those positions    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    kw = {k: v.to(device) for k, v in kw.items()}
    return x, y, kw

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
    mask_token_id = meta_vocab_size   # one extra id after the real codes (8192 for EEG); used when bidirectional
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

#jm unigram baseline: bits per token of a model that only knows how often each token occurs (val split)
_val = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
_p = np.bincount(_val, minlength=meta_vocab_size or 0).astype(np.float64) / len(_val)
_p = _p[_p > 0]
unigram_bits = float(-(_p * np.log2(_p)).sum())
config['unigram_bits'] = unigram_bits   # stored with the wandb run
print(f"unigram baseline: {unigram_bits:.2f} bits per token")

# model init
codebook = None
if use_codebook: #jm
    codebook = torch.from_numpy(np.load(os.path.join(data_dir, 'codebook.npy'))).float()   # (8192, 8)
    codebook = torch.nn.functional.normalize(codebook, dim=1)   # unit length: the DAC compares codes by direction
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, #jm
                  bias=bias, vocab_size=None, dropout=dropout,
                  codebook_dim=codebook.shape[1] if codebook is not None else 0,
                  pos_encoding=pos_encoding,
                  bidirectional=bidirectional,
                  channel_encoding=channel_encoding) #jm
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = (meta_vocab_size if meta_vocab_size is not None else 50304) + (1 if bidirectional else 0) #jm
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    if codebook is not None: #jm
        model.codebook[:codebook.shape[0]].copy_(codebook)   # the MASK row (last) stays zero
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
codebook_dev = codebook.to(device) if codebook is not None else None   #jm (8192, 8) unit vectors, for the eval L2 metric

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
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
# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y, kw = get_batch(split)
#             with ctx:
#                 logits, loss = model(X, Y, **kw)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out

@torch.no_grad()
def token_metrics(logits, Y):   #jm per-token metrics over the predicted positions (Y != -1); used for train steps and eval
    tgt = Y.reshape(-1)
    keep = tgt != -1
    lg = logits.reshape(-1, logits.size(-1))[keep][:, :meta_vocab_size].float()   # never count MASK as a prediction
    logp = torch.log_softmax(lg, dim=-1)
    true = tgt[keep]
    pred = lg.argmax(-1)
    m = {'top1': (pred == true).float().mean().item(),                                # most likely code is the true one
         'top5': (lg.topk(5, dim=-1).indices == true[:, None]).any(-1).float().mean().item(),   # true code among the 5 most likely
         'entropy': -(logp.exp() * logp).sum(-1).mean().item(),                       # spread of the prediction (nats)
         'conf': logp.max(-1).values.exp().mean().item(),                              # probability of the favourite
         'conf_nll': -logp.max(-1).values.mean().item()}                               # -ln p(favourite), loss scale
    if codebook_dev is not None:
        m['l2_code'] = (codebook_dev[pred] - codebook_dev[true]).norm(dim=-1).mean().item()   # DAC 8-dim distance, 0 = exact
    return m

@torch.no_grad()
def estimate_loss():   #jm validation only (training metrics are logged every step from the training batch)
    model.eval()
    losses, metrics = [], []
    for k in range(eval_iters):
        X, Y, kw = get_batch('val')
        with ctx:
            logits, loss = model(X, Y, **kw)
        losses.append(loss.item())
        metrics.append(token_metrics(logits, Y))
    model.train()
    out = {'val': float(np.mean(losses))}
    out.update({'val_' + key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]})
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
    n_params = sum(p.numel() for p in model.parameters())
    config['n_params'] = n_params                                    #jm shows up as a config column in wandb
    notes = (f"{n_layer} layers x {n_head} heads x {n_embd} dim, {n_params/1e6:.2f}M params; "
             f"dataset {dataset}, block {block_size}, batch {batch_size}, {max_iters} iters; "
             f"mask {mask_ratio if bidirectional else 'none (next-token)'}, channels {channel_encoding}, "
             f"time {pos_encoding}, codebook input {use_codebook}")
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, notes=notes)

# training loop
X, Y, kw = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({   #jm step=iter_num: wandb's x-axis is the training iteration
                "val/loss": losses['val'],
                "val/bits_saved": unigram_bits - losses['val'] / math.log(2),
                "val/perplexity": math.exp(losses['val']),
                "val/effective_codes": math.exp(losses['val_entropy']),
                **{f"val/{k[4:]}": v for k, v in losses.items() if k != 'val'},   # val/top1, val/top5, val/l2_code, val/entropy, val/conf, val/conf_nll
            }, step=iter_num)
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
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
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
            logits, loss = model(X, Y, **kw)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        if micro_step == gradient_accumulation_steps - 1 and iter_num % log_interval == 0: #jm metrics of the batch just trained on
            step_metrics = token_metrics(logits, Y)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, kw = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    grad_norm = float('nan') #jm
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item() #jm norm before clipping
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
        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            wandb.log({"train/loss": lossf, "train/perplexity": math.exp(lossf), "train/grad_norm": grad_norm, "lr": lr, #jm
                       **{f"train/{k}": v for k, v in step_metrics.items() if k != 'conf_nll'}}, step=iter_num)   #jm conf_nll: val only
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
