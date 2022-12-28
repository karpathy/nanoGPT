"""
Train a GPT model on a dataset of text. One GPU version.
The text is assumed to pre-tokenized and inside files train.pt and val.pt
"""

import os
import time
import math

import numpy as np
import torch
import wandb

from model import GPTConfig, GPT
# -----------------------------------------------------------------------------
# settings, todo argparse or something
# I/O
out_dir = 'out'
eval_interval = 500
log_interval = 1
# wandb logging
wandb_log = False # disabled by default
wandb_entity = 'karpathy'
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
batch_size = 8
block_size = 1024
# model
device = 'cuda:0'
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
dropout = 0.1
n_layer = 12
n_head = 12
n_embd = 768
# adamw optimizer
learning_rate = 2.5e-4 # max learning rate
max_iters = 500000 # total number of training iterations
weight_decay = 1e-2
betas = (0.9, 0.95)
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 320000 # how many steps to decay the learning rate for
min_lr = 1e-5 # minimum learning rate
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# poor man's data loader, TODO use real DataLoader...
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# model init
# TODO I don't love this whole part/API yet
model_args = dict(n_layer = n_layer, n_head = n_head, n_embd = n_embd, block_size = block_size, dropout = dropout)
if init_from == 'scratch':
    # init a new model from scratch
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # resume training from a checkpoint. TODO: do we resume iter_num etc too? (yes...)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path)
    checkpoint_model_args = checkpoint['model_args']
    for k, v in model_args.items():
        assert checkpoint_model_args[k] == v, "for now"
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
elif init_from.startswith('gpt2'):
    # initialize from OpenAI GPT-2 weights
    model = GPT.from_pretrained(init_from)
    if block_size < model.block_size:
        model.crop_block_size(block_size)
model.to(device)

@torch.no_grad()
def estimate_loss(eval_iters=50):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, betas)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# learning rate decay scheduler (cosine with warmup)
def get_lr(iter):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) if iter > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name)
    wandb.config = {
        "batch_size": batch_size,
        "block_size": block_size,
        "learning_rate": learning_rate, # TODO log everything else too
    }

# training loop
iter_num = 0
num_tokens = 0
best_val_loss = 1e9
t0 = time.time()
while True:

    # determine the learning rate for this iteration
    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = learning_rate

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "num_tokens": num_tokens,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0: # don't save checkpoints on very first iteration...
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    X, Y = get_batch('train')
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(X, Y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # TODO: gradient clipping
    optimizer.step()

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() # loss as float. TODO CPU-GPU sync: profile, make sure not slow af
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    num_tokens += X.numel()

    # termination conditions
    if iter_num >= max_iters:
        break

