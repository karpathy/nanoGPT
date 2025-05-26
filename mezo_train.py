"""
This training script implements MeZO (Memory-efficient Zero-Order) optimization for nanoGPT.
Based on the original nanoGPT training script but uses zeroth-order optimization.

To run on a single GPU, example:
$ python mezo_train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 mezo_train.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import psutil
import torch.cuda as cuda
from tqdm import tqdm

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out-mezo'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2-mezo' # 'run' + str(time.time())
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
# MeZO specific parameters
zo_eps = 1e-3       # perturbation size for zero-order gradient estimation
zo_by_layer = False # whether to update each layer separately (True) or all at once (False)
zo_q = 1            # number of finite differences computations to average over
use_momentum = False # disable momentum for standard MeZO
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
if not os.path.exists(os.path.join(data_dir, 'train.bin')):
    data_dir = '/nobackups/allanath/data-nanogpt'  # fallback location

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# MeZO specific functions
# -----------------------------------------------------------------------------

# Function to perturb model parameters with Gaussian perturbations
def zo_perturb_parameters(model, zo_random_seed, scaling_factor=1, layer_idx=None):
    """
    Perturb the parameters with random vector z.
    Optionally perturb only a specific layer if layer_idx is provided.
    """
    torch.manual_seed(zo_random_seed)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # If layer_idx is specified, only perturb that layer
            if zo_by_layer and layer_idx is not None:
                # Extract layer number from parameter name (e.g., 'transformer.h.3.mlp.c_fc.weight')
                try:
                    layer_num = int(name.split('.h.')[1].split('.')[0])
                    if layer_num != layer_idx:
                        continue
                except (IndexError, ValueError):
                    # Not a transformer layer parameter, skip
                    continue
            
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data = param.data + scaling_factor * z * zo_eps

# Function to perform forward pass without gradients
def zo_forward(model, x, y):
    """
    Get (no gradient) loss from the model.
    """
    model.eval()
    with torch.inference_mode():
        with ctx:
            logits, loss = model(x, y)
    return loss.detach()

# Function to estimate gradient using MeZO with multiple (q) estimates
def zo_step(model, x, y, layer_idx=None):
    """
    Estimate gradient using MeZO with q different perturbations.
    For q>1, directly accumulates the gradient estimates.
    """
    # Initialize dictionary to store the accumulated gradient for each parameter
    accumulated_grads = {}
    
    # Initialize parameters we track
    first_loss = None
    
    # For each perturbation
    for q_idx in range(zo_q):
        # Sample random seed for deterministic perturbation
        zo_random_seed = np.random.randint(1000000000)
        torch.manual_seed(zo_random_seed)
        
        # Generate perturbation z for all relevant parameters
        perturbation_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Filter by layer if needed
                if zo_by_layer and layer_idx is not None:
                    try:
                        layer_num = int(name.split('.h.')[1].split('.')[0])
                        if layer_num != layer_idx:
                            continue
                    except (IndexError, ValueError):
                        continue
                
                # Generate z ~ N(0, I)
                z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                perturbation_dict[name] = z
        
        # Apply positive perturbation: θ + ε*z
        for name, param in model.named_parameters():
            if param.requires_grad and name in perturbation_dict:
                param.data = param.data + zo_eps * perturbation_dict[name]
        
        # Evaluate f(θ + ε*z)
        f_plus = zo_forward(model, x, y)
        
        # Save first loss from first perturbation for reporting
        if q_idx == 0:
            first_loss = f_plus
        
        # Apply negative perturbation: θ - ε*z (which is θ - 2ε*z from current state)
        for name, param in model.named_parameters():
            if param.requires_grad and name in perturbation_dict:
                param.data = param.data - 2 * zo_eps * perturbation_dict[name]
        
        # Evaluate f(θ - ε*z)
        f_minus = zo_forward(model, x, y)
        
        # Compute scalar coefficient: (f_plus - f_minus) / (2*ε)
        coeff = (f_plus - f_minus).item() / (2 * zo_eps)
        
        # Accumulate gradient estimate for each parameter: g += (f_plus - f_minus)/(2*ε) * z
        for name, param in model.named_parameters():
            if param.requires_grad and name in perturbation_dict:
                if name not in accumulated_grads:
                    accumulated_grads[name] = coeff * perturbation_dict[name]
                else:
                    accumulated_grads[name] += coeff * perturbation_dict[name]
        
        # Reset parameters back to original values
        for name, param in model.named_parameters():
            if param.requires_grad and name in perturbation_dict:
                param.data = param.data + zo_eps * perturbation_dict[name]  # θ - ε*z + ε*z = θ
    
    # Average the accumulated gradients if q > 1
    if zo_q > 1:
        for name in accumulated_grads:
            accumulated_grads[name] = accumulated_grads[name] / zo_q
    
    # Return the first loss and the accumulated gradients
    return first_loss, accumulated_grads

# Function to update parameters using MeZO
def zo_update(model, optimizer, gradient_info, lr, layer_idx=None):
    """
    Update model parameters using the MeZO gradient estimate.
    """
    # Unpack the gradient information (dictionary of parameter gradients)
    grad_dict = gradient_info
    
    # Apply the gradient update to each parameter
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Skip parameters not in gradient dictionary (filtered by layer)
            if name not in grad_dict:
                continue
            
            # Get the gradient for this parameter
            grad = grad_dict[name]
            
            # Apply update with or without weight decay
            is_weight = "bias" not in name and "layer_norm" not in name and "layernorm" not in name
            if is_weight:
                param.data = param.data - lr * (grad + weight_decay * param.data)
            else:
                param.data = param.data - lr * grad

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

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume' and 'optimizer' in checkpoint:
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
                logits, loss = model(X, Y)
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

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# Create progress bar
pbar = tqdm(total=max_iters, desc='Training', disable=not master_process)

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"\nstep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
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

    # forward backward update with MeZO optimization
    for micro_step in range(gradient_accumulation_steps):
        if zo_by_layer:
            # For layer-wise MeZO, we need to estimate and update gradients for each layer separately
            # Get total number of layers
            num_layers = model_args['n_layer']
            
            # Store first loss for logging
            first_loss = None
            
            # Update each layer one by one
            for layer_idx in range(num_layers):
                loss, grad_dict = zo_step(model, X, Y, layer_idx=layer_idx)
                
                # Store the first loss for logging
                if layer_idx == 0 and micro_step == 0:
                    first_loss = loss
                
                # Scale the gradients appropriately
                for name in grad_dict:
                    grad_dict[name] = grad_dict[name] / gradient_accumulation_steps
                
                # Update parameters for this layer immediately
                zo_update(raw_model, optimizer, grad_dict, lr, layer_idx=layer_idx)
            
            if first_loss is not None:
                loss = first_loss  # Use the first loss for logging purposes
                
        else:
            # Standard MeZO: update all parameters at once
            loss, grad_dict = zo_step(model, X, Y)
            
            # Scale the loss to account for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Accumulate the gradients
            if micro_step == 0:
                accumulated_grads = {}
                for name, grad in grad_dict.items():
                    accumulated_grads[name] = grad / gradient_accumulation_steps
            else:
                for name, grad in grad_dict.items():
                    accumulated_grads[name] += grad / gradient_accumulation_steps
                
        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        
    # Update parameters using MeZO (only for non-layer-wise mode)
    if not zo_by_layer:
        zo_update(raw_model, optimizer, accumulated_grads, lr)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        # Get memory usage
        if device_type == 'cuda':
            memory_allocated = cuda.memory_allocated() / 1e9  # Convert to GB
            memory_reserved = cuda.memory_reserved() / 1e9    # Convert to GB
        else:
            process = psutil.Process()
            memory_allocated = process.memory_info().rss / 1e9  # Convert to GB
            memory_reserved = 0
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{lossf:.4f}',
            'lr': f'{lr:.2e}',
            'mem': f'{memory_allocated:.1f}GB',
            'mfu': f'{running_mfu*100:.1f}%'
        })
        pbar.update(1)
    
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

pbar.close()
if ddp:
    destroy_process_group() 