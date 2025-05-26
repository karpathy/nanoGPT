"""
Gradient Rank Analysis for nanoGPT training.
This script trains a model with SGD while analyzing the effective rank of gradient matrices
for each layer to understand the intrinsic dimensionality of gradients.

Usage:
$ python gradient_rank_analysis.py --config=config/gradient_rank_analysis.py
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
import json
from collections import defaultdict

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration (similar to main training script)
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out-gradient-rank-analysis'
eval_interval = 500
log_interval = 50
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

# model (small for quick analysis)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# SGD optimizer
learning_rate = 1e-3
max_iters = 2000  # Shorter for analysis
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 200
lr_decay_iters = 2000
min_lr = 1e-4

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
train_method = 'sgd'  # Force SGD for gradient analysis

# Gradient rank analysis specific parameters
rank_analysis_interval = 25  # Analyze every N steps
svd_tau_thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]  # Different thresholds for effective rank
save_singular_values = True  # Whether to save full singular value distributions
max_rank_to_analyze = 100  # Maximum rank to compute (for efficiency)

# -----------------------------------------------------------------------------
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# Setup similar to main training script
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
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
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loading
data_dir = os.path.join('data', dataset)
if not os.path.exists(os.path.join(data_dir, 'train.bin')):
    data_dir = '/nobackups/allanath/data-nanogpt'

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

# -----------------------------------------------------------------------------
# Gradient Rank Analysis Functions
# -----------------------------------------------------------------------------

def compute_effective_rank(singular_values, tau=0.1):
    """
    Compute effective rank using different methods.
    
    Args:
        singular_values: 1D tensor of singular values (sorted in descending order)
        tau: Threshold for counting significant singular values
    
    Returns:
        dict with different rank measures
    """
    if len(singular_values) == 0:
        return {
            'threshold_rank': 0,
            'entropy_rank': 0,
            'stable_rank': 0,
            'num_components': 0
        }
    
    # Threshold-based effective rank
    sigma_max = singular_values[0]
    threshold_rank = (singular_values >= tau * sigma_max).sum().item()
    
    # Entropy-based effective rank (numerical stability)
    sigma_squared = singular_values ** 2
    sigma_squared_norm = sigma_squared / sigma_squared.sum()
    sigma_squared_norm = torch.clamp(sigma_squared_norm, min=1e-12)  # Avoid log(0)
    entropy = -(sigma_squared_norm * torch.log(sigma_squared_norm)).sum()
    entropy_rank = torch.exp(entropy).item()
    
    # Stable rank (Frobenius norm / spectral norm)
    stable_rank = (sigma_squared.sum() / (sigma_max ** 2)).item()
    
    return {
        'threshold_rank': threshold_rank,
        'entropy_rank': entropy_rank,
        'stable_rank': stable_rank,
        'num_components': len(singular_values)
    }

def analyze_gradient_rank(model, step):
    """
    Analyze the effective rank of gradient matrices for each layer.
    
    Args:
        model: The model with computed gradients
        step: Current training step
    
    Returns:
        dict: Analysis results for each layer
    """
    analysis_results = {}
    
    for name, param in model.named_parameters():
        if param.grad is None or param.ndim < 2:
            continue  # Skip parameters without gradients or 1D parameters
        
        # Clean parameter name
        clean_name = name
        if name.startswith('_orig_mod.'):
            clean_name = name[len('_orig_mod.'):]
        
        grad = param.grad.detach()
        
        # For efficiency, limit the rank we analyze
        max_rank = min(max_rank_to_analyze, min(grad.shape))
        
        # Compute SVD (use economy SVD for efficiency)
        try:
            U, sigma, Vt = torch.linalg.svd(grad, full_matrices=False)
            # Truncate to max_rank for efficiency
            sigma = sigma[:max_rank]
        except Exception as e:
            print(f"SVD failed for {clean_name}: {e}")
            continue
        
        # Analyze effective rank with different thresholds
        layer_analysis = {
            'step': step,
            'shape': list(grad.shape),
            'max_possible_rank': min(grad.shape),
            'singular_values': sigma.cpu().numpy().tolist() if save_singular_values else None,
            'rank_measures': {}
        }
        
        # Compute effective rank with different tau values
        for tau in svd_tau_thresholds:
            rank_info = compute_effective_rank(sigma, tau)
            layer_analysis['rank_measures'][f'tau_{tau}'] = rank_info
        
        analysis_results[clean_name] = layer_analysis
    
    return analysis_results

def save_analysis_results(results, step):
    """Save analysis results to files."""
    if not master_process:
        return
    
    # Save detailed results as JSON
    results_file = os.path.join(out_dir, f'gradient_rank_analysis_step_{step}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also append summary to a CSV-like file for easy plotting
    summary_file = os.path.join(out_dir, 'gradient_rank_summary.csv')
    write_header = not os.path.exists(summary_file)
    
    with open(summary_file, 'a') as f:
        if write_header:
            # Write header
            header = ['step', 'layer_name', 'layer_type', 'shape', 'max_rank']
            for tau in svd_tau_thresholds:
                header.extend([f'threshold_rank_tau_{tau}', f'entropy_rank_tau_{tau}', f'stable_rank_tau_{tau}'])
            f.write(','.join(header) + '\n')
        
        # Write data for each layer
        for layer_name, analysis in results.items():
            layer_type = 'unknown'
            if 'attn' in layer_name and 'proj' in layer_name:
                layer_type = 'attention_proj'
            elif 'attn' in layer_name:
                layer_type = 'attention'
            elif 'mlp' in layer_name:
                layer_type = 'mlp'
            elif 'wte' in layer_name or 'wpe' in layer_name or 'embed' in layer_name:
                layer_type = 'embedding'
            elif 'head' in layer_name:
                layer_type = 'output_head'
            
            row = [
                str(step),
                layer_name,
                layer_type,
                f"({analysis['shape'][0]}x{analysis['shape'][1]})",
                str(analysis['max_possible_rank'])
            ]
            
            for tau in svd_tau_thresholds:
                rank_measures = analysis['rank_measures'][f'tau_{tau}']
                row.extend([
                    str(rank_measures['threshold_rank']),
                    f"{rank_measures['entropy_rank']:.2f}",
                    f"{rank_measures['stable_rank']:.2f}"
                ])
            
            f.write(','.join(row) + '\n')

# -----------------------------------------------------------------------------
# Model initialization and training setup
# -----------------------------------------------------------------------------

# Model setup (similar to main script)
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    if master_process:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if master_process:
    print("Initializing a new model from scratch for gradient rank analysis")

if meta_vocab_size is None:
    if master_process:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

if master_process:
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# Initialize optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Compile model
if compile:
    if master_process:
        print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# Wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Learning rate decay scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

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

# -----------------------------------------------------------------------------
# Training loop with gradient rank analysis
# -----------------------------------------------------------------------------

iter_num = 0
best_val_loss = 1e9
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
raw_model = model.module if ddp else model

if master_process:
    print(f"Starting gradient rank analysis training for {max_iters} iterations")
    print(f"Rank analysis will be performed every {rank_analysis_interval} steps")

while True:
    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    
    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"\nstep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward-backward pass with gradient accumulation
    model.train()
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Immediately async prefetch next batch
        X, Y = get_batch('train')
    
    # Gradient rank analysis (before clipping and optimizer step)
    if iter_num % rank_analysis_interval == 0:
        if master_process:
            print(f"\nPerforming gradient rank analysis at step {iter_num}...")
        
        analysis_results = analyze_gradient_rank(raw_model, iter_num)
        save_analysis_results(analysis_results, iter_num)
        
        if master_process:
            # Print a quick summary
            total_layers = len(analysis_results)
            avg_entropy_rank = np.mean([
                analysis['rank_measures']['tau_0.1']['entropy_rank'] 
                for analysis in analysis_results.values()
            ])
            avg_threshold_rank = np.mean([
                analysis['rank_measures']['tau_0.1']['threshold_rank'] 
                for analysis in analysis_results.values()
            ])
            print(f"  Analyzed {total_layers} layers")
            print(f"  Average entropy rank (τ=0.1): {avg_entropy_rank:.1f}")
            print(f"  Average threshold rank (τ=0.1): {avg_threshold_rank:.1f}")
    
    # Clip gradients
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Step the optimizer
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    
    iter_num += 1
    local_iter_num += 1
    
    # Termination condition
    if iter_num > max_iters:
        break

if master_process:
    print(f"\nGradient rank analysis complete!")
    print(f"Results saved in: {out_dir}")
    print(f"Summary CSV: {os.path.join(out_dir, 'gradient_rank_summary.csv')}")

if ddp:
    destroy_process_group() 