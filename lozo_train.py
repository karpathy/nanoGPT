"""
This training script implements LOZO (Low-Rank Zero-Order) optimization for nanoGPT.
Based on the original nanoGPT training script but uses zeroth-order optimization with low-rank perturbations.

To run on a single GPU, example:
$ python lozo_train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 lozo_train.py
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
out_dir = 'out-lozo'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2-lozo' # 'run' + str(time.time())
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
# LOZO specific parameters
zo_eps = 1e-3       # perturbation size for zero-order gradient estimation
rank_r = 4          # rank for low-rank perturbation
step_interval = 10  # interval for updating the V matrices (every ν steps)
use_momentum = False # whether to use momentum in LOZO (LOZO-M)
momentum_beta = 0.9 # momentum coefficient for LOZO-M
zo_q = 1            # number of finite differences computations to average over
# Rank-adaptive LOZO parameters
rank_adaptive = False # whether to use adaptive rank scheduling
min_rank = 1        # minimum rank for adaptive scheduling
max_rank = 16       # maximum rank for adaptive scheduling
# SVD-LOZO specific parameters
svd_tau = 0.6       # threshold for adaptive rank selection (keep σ > τ * σ_max) - increased for actual rank reduction
svd_max_rank = 16   # maximum rank for randomized SVD
use_full_svd = False # whether to use full SVD (True) or randomized SVD (False)
# Training method - 'lozo', 'lozom', 'svdlozo', 'mezo', 'mezom', 'dimezo', 'dilozo', 'kronzo', 'adam', 'sgd'
train_method = 'lozo'
# DiMeZO specific parameters
directional_q = 10       # number of directions to try in directional selection
dimezo_direct_movement = False  # if True, move directly in best direction; if False, use gradient estimation (f+ - f-)/2ε
# KronZO specific parameters
kron_max_factor = 32     # maximum factor size for Kronecker factorization
kron_strategy = 'approx_square'  # 'approx_square', 'fixed_factor', 'power2'
# Adaptive zo_eps parameters
use_adaptive_eps = False    # whether to use adaptive zo_eps
adaptive_eps_window = 20    # window size for tracking success rate
adaptive_eps_lr_coupling = 0.5  # strength of coupling between eps and learning rate (0=none, 1=full)
adaptive_eps_success_high = 0.7  # success rate threshold for increasing eps
adaptive_eps_success_low = 0.3   # success rate threshold for decreasing eps
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
# LOZO specific functions
# -----------------------------------------------------------------------------

def get_current_rank(iter_num, max_iters, min_rank, max_rank):
    """
    Compute current rank using linear scheduling from min_rank to max_rank.
    
    Args:
        iter_num: Current iteration number
        max_iters: Total number of iterations
        min_rank: Starting rank
        max_rank: Ending rank
    
    Returns:
        current_rank: Integer rank for current iteration
    """
    if not rank_adaptive:
        return rank_r  # Use fixed rank if adaptive is disabled
    
    # Linear interpolation from min_rank to max_rank
    progress = min(iter_num / max_iters, 1.0)  # Clamp to [0, 1]
    current_rank_float = min_rank + progress * (max_rank - min_rank)
    current_rank = max(min_rank, min(max_rank, int(round(current_rank_float))))
    
    return current_rank

# Function to perturb model parameters with low-rank perturbations
def lowrank_zo_perturb_parameters(model, v_dict, zo_random_seed, step, scaling_factor=1, current_rank=None):
    """
    Perturb the parameters with random vector uv^t.
    Only update V matrices every step_interval steps.
    Supports adaptive rank by resizing V matrices when needed.
    """
    if current_rank is None:
        current_rank = rank_r  # Use default rank if not specified
        
    torch.manual_seed(zo_random_seed)
    
    # Create a list of named parameters to optimize - refresh on each call to handle compiled models
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Handle _orig_mod prefix from torch.compile
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))
    
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # For matrices, use low-rank perturbation
            # Create new V matrix at the specified interval or if not present
            need_new_v = (step % step_interval == 0 or 
                         clean_name not in v_dict or 
                         v_dict[clean_name].size(1) != current_rank)  # Rank changed
            
            if need_new_v:
                v = torch.randn(param.size(1), current_rank, device=param.device, dtype=param.dtype)
                v_dict[clean_name] = v
            else:
                v = v_dict[clean_name]
            
            u = torch.randn(param.size(0), current_rank, device=param.device, dtype=param.dtype)
            param.data = param.data + scaling_factor * (u @ v.t()) * zo_eps
        else:
            # For vectors (biases), use Gaussian perturbation
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data = param.data + scaling_factor * z * zo_eps
    
    return named_parameters_to_optim

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

# Function to estimate gradient using LOZO with multiple (q) estimates
def lowrank_zo_step(model, X, Y, v_dict, step, zo_random_seed, current_rank=None):
    """
    Estimate gradient using LOZO with q-times finite differences.
    
    For q>1:
    1. Initialize g = 0
    2. For each q iteration:
       - Generate a single UV^T low-rank perturbation
       - Compute (f+ - f-)/2*zo_eps
       - Accumulate: g += (scalar coefficient) * UV^T
    3. Average: g = g / q
    """
    if current_rank is None:
        current_rank = rank_r  # Use default rank if not specified
        
    # If q=1, use the original implementation for maximum compatibility
    if zo_q == 1:
        # First function evaluation
        lowrank_zo_perturb_parameters(model, v_dict, zo_random_seed, step, scaling_factor=1, current_rank=current_rank)
        loss1 = zo_forward(model, X, Y)

        # Second function evaluation
        lowrank_zo_perturb_parameters(model, v_dict, zo_random_seed, step, scaling_factor=-2, current_rank=current_rank)
        loss2 = zo_forward(model, X, Y)

        # Calculate projected gradient
        projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()

        # Reset model back to its parameters at start of step
        lowrank_zo_perturb_parameters(model, v_dict, zo_random_seed, step, scaling_factor=1, current_rank=current_rank)
        
        return loss1, projected_grad, zo_random_seed
    
    # For q>1, implement the multiple estimation approach
    # Initialize an empty gradient accumulator for each parameter
    accumulated_grads = {}
    first_loss = None
    
    # Create a list of parameters to optimize
    params_to_optimize = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            params_to_optimize.append((clean_name, param))
    
    # For each q estimation
    for q_idx in range(zo_q):
        # Sample a new random seed for this estimation
        current_zo_seed = np.random.randint(1000000000) if q_idx > 0 else zo_random_seed
        torch.manual_seed(current_zo_seed)
        
        # Generate perturbations for each parameter
        perturbations = {}  # Store the perturbations for this iteration
        
        for clean_name, param in params_to_optimize:
            if param.ndim >= 2:
                # For matrices, use low-rank perturbation
                if clean_name in v_dict:
                    v = v_dict[clean_name]
                    u = torch.randn(param.size(0), current_rank, device=param.device, dtype=param.dtype)
                    perturbation = u @ v.t()
                    perturbations[clean_name] = (perturbation, True)  # (perturbation, is_lowrank)
                    
                    # Apply positive perturbation
                    param.data = param.data + perturbation * zo_eps
                else:
                    # For matrices without v vector, use Gaussian
                    z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                    perturbations[clean_name] = (z, False)
                    
                    # Apply positive perturbation
                    param.data = param.data + z * zo_eps
            else:
                # For vectors, use Gaussian perturbation
                z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                perturbations[clean_name] = (z, False)
                
                # Apply positive perturbation
                param.data = param.data + z * zo_eps
        
        # Evaluate f(θ + perturbation)
        f_plus = zo_forward(model, X, Y)
        
        # Save first loss for reporting
        if q_idx == 0:
            first_loss = f_plus
        
        # Apply negative perturbation (from current state)
        for clean_name, param in params_to_optimize:
            if clean_name in perturbations:
                perturbation, _ = perturbations[clean_name]
                param.data = param.data - 2 * perturbation * zo_eps
        
        # Evaluate f(θ - perturbation)
        f_minus = zo_forward(model, X, Y)
        
        # Compute scalar coefficient
        coeff = (f_plus - f_minus).item() / (2 * zo_eps)
        
        # Accumulate gradients for each parameter
        for clean_name, param in params_to_optimize:
            if clean_name in perturbations:
                perturbation, is_lowrank = perturbations[clean_name]
                
                # Calculate gradient for this parameter
                grad = coeff * perturbation
                
                # Add to accumulated gradients
                if clean_name not in accumulated_grads:
                    accumulated_grads[clean_name] = grad
                else:
                    accumulated_grads[clean_name] += grad
        
        # Reset parameters to their original values
        for clean_name, param in params_to_optimize:
            if clean_name in perturbations:
                perturbation, _ = perturbations[clean_name]
                param.data = param.data + perturbation * zo_eps
    
    # Average the gradients if q > 1
    if zo_q > 1:
        for clean_name in accumulated_grads:
            accumulated_grads[clean_name] = accumulated_grads[clean_name] / zo_q
    
    # Return the first loss and accumulated gradients
    return first_loss, accumulated_grads, zo_random_seed

# Function to update parameters using LOZO (original version for q=1)
def lowrank_zo_update(model, optimizer, projected_grad, zo_random_seed, v_dict, step, lr, named_parameters_to_optim=None, current_rank=None):
    """
    Update model parameters using the LOZO gradient estimate.
    """
    if current_rank is None:
        current_rank = rank_r  # Use default rank if not specified
        
    torch.manual_seed(zo_random_seed)
    
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Debug logging only on first step
    if step == 0 and master_process:
        print(f"v_dict contains {len(v_dict)} parameter entries")
        print(f"First few keys: {list(v_dict.keys())[:5]}")
        print(f"Total parameters to optimize: {len(named_parameters_to_optim)}")
    
    missing_params = []
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # For matrices, use low-rank update
            if clean_name not in v_dict:
                missing_params.append(clean_name)
                continue
                
            v = v_dict[clean_name]
            u = torch.randn(param.size(0), current_rank, device=param.device, dtype=param.dtype)
            
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (projected_grad * (u @ v.t()) + weight_decay * param.data)
            else:
                param.data = param.data - lr * (projected_grad * (u @ v.t()))
        else:
            # For vectors (biases), use Gaussian update
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (projected_grad * z + weight_decay * param.data)
            else:
                param.data = param.data - lr * (projected_grad * z)
    
    # Only print missing parameters once
    if step == 0 and missing_params and len(missing_params) > 0 and master_process:
        print(f"Missing {len(missing_params)}/{len(named_parameters_to_optim)} parameters in v_dict")
        print(f"First few missing: {missing_params[:5]}")

# Function to update parameters using LOZO with direct gradient dictionary
def lowrank_zo_update_direct(model, optimizer, grad_dict, lr, named_parameters_to_optim=None):
    """
    Update model parameters using the pre-computed gradient dictionary from q-times estimation.
    """
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Apply the gradient update to each parameter
    for clean_name, param in named_parameters_to_optim:
        # Skip parameters not in gradient dictionary
        if clean_name not in grad_dict:
            continue
        
        # Get the gradient for this parameter
        grad = grad_dict[clean_name]
        
        # Apply update with or without weight decay
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        if is_weight:
            param.data = param.data - lr * (grad + weight_decay * param.data)
        else:
            param.data = param.data - lr * grad

# Function to update parameters using LOZO with momentum (LOZO-M)
def lowrank_zo_update_momentum(model, optimizer, projected_grad, zo_random_seed, v_dict, exp_avg_m, v_old_dict, step, lr, beta1=0.9, named_parameters_to_optim=None, current_rank=None):
    """
    Update model parameters using the LOZO gradient estimate with momentum.
    This is more memory-efficient than standard momentum and can lead to better performance.
    
    Args:
        model: The model to update
        optimizer: The optimizer (not directly used but kept for API consistency)
        projected_grad: The projected gradient estimate from LOZO
        zo_random_seed: Random seed for reproducibility
        v_dict: Dictionary of V matrices for low-rank updates
        exp_avg_m: Dictionary of exponential moving average for momentum
        v_old_dict: Dictionary of old V matrices (from previous step_interval)
        step: Current optimization step
        lr: Learning rate
        beta1: Momentum coefficient (default: 0.9)
        named_parameters_to_optim: List of (name, parameter) tuples to optimize
        current_rank: Current rank for adaptive rank scheduling
    """
    if current_rank is None:
        current_rank = rank_r  # Use default rank if not specified
        
    torch.manual_seed(zo_random_seed)     
    
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Debug logs on first step
    if step == 0 and master_process:
        print(f"v_dict contains {len(v_dict)} parameter entries for LOZO-M")
        print(f"First few keys: {list(v_dict.keys())[:5]}")
        print(f"Total parameters to optimize: {len(named_parameters_to_optim)}")
    
    missing_params = []
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # For matrices, use low-rank update with momentum
            if clean_name not in v_dict:
                missing_params.append(clean_name)
                continue
                
            v = v_dict[clean_name]
            u = torch.randn(param.size(0), current_rank, device=param.device, dtype=param.dtype)
            
            # Initialize momentum if needed
            if clean_name not in exp_avg_m:
                exp_avg_m[clean_name] = torch.zeros_like(u)
            
            # Update momentum based on step interval
            if step % step_interval == 0:
                if clean_name in v_old_dict:   
                    # Use the transition matrix between old and new V
                    v_old = v_old_dict[clean_name]
                    n = v_old.shape[0]  # Use row dimension as in original LOZO
                    exp_avg_m[clean_name] = beta1 * (exp_avg_m[clean_name] @ v_old.t() @ v / n) + (1 - beta1) * projected_grad * u
                else:
                    # First initialization or no old V available
                    exp_avg_m[clean_name] = projected_grad * u
            elif step % step_interval == step_interval - 1:
                # Store old V matrix before it changes in next step
                v_old_dict[clean_name] = v  # No need for clone() as we only read from it
                exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * projected_grad * u
            else:
                # Regular momentum update
                exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * projected_grad * u
            
            # Apply update
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (exp_avg_m[clean_name] @ v.t() + weight_decay * param.data)
            else:
                param.data = param.data - lr * (exp_avg_m[clean_name] @ v.t())
        else:
            # For vectors, use Gaussian update with momentum
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            
            # Initialize or update momentum
            if clean_name not in exp_avg_m:
                exp_avg_m[clean_name] = projected_grad * z
            else:
                exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * projected_grad * z
            
            # Apply update
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (exp_avg_m[clean_name] + weight_decay * param.data)
            else:
                param.data = param.data - lr * exp_avg_m[clean_name]
    
    # Only print missing parameters once
    if step == 0 and missing_params and len(missing_params) > 0 and master_process:
        print(f"Missing {len(missing_params)}/{len(named_parameters_to_optim)} parameters in v_dict")
        print(f"First few missing: {missing_params[:5]}")

# -----------------------------------------------------------------------------
# SVD-LOZO specific functions
# -----------------------------------------------------------------------------

def randomized_svd(A, n_components, n_oversamples=10):
    """
    Compute randomized SVD for efficient approximate decomposition.
    
    Args:
        A: Input matrix (m x n)
        n_components: Number of components to compute
        n_oversamples: Additional samples for better approximation
        
    Returns:
        U, sigma, Vt: SVD components where A ≈ U @ diag(sigma) @ Vt
    """
    m, n = A.shape
    k = min(n_components + n_oversamples, min(m, n))
    
    # Random projection
    Omega = torch.randn(n, k, device=A.device, dtype=A.dtype)
    Y = A @ Omega  # Shape: [m, k]
    
    # QR decomposition - use new API
    Q, _ = torch.linalg.qr(Y, mode='reduced')  # Q shape: [m, k]
    
    # Project and compute SVD of smaller matrix
    B = Q.T @ A  # Shape: [k, n]
    U_hat, sigma, Vt = torch.linalg.svd(B, full_matrices=False)
    
    # Recover U
    U = Q @ U_hat  # Shape: [m, k]
    
    # Truncate to desired number of components
    n_components = min(n_components, len(sigma))
    return U[:, :n_components], sigma[:n_components], Vt[:n_components, :]

def full_svd(A, n_components):
    """
    Compute full SVD and truncate to n_components.
    More accurate but computationally expensive than randomized SVD.
    
    Args:
        A: Input matrix (m x n)
        n_components: Number of components to compute
        
    Returns:
        U, sigma, Vt: SVD components where A ≈ U @ diag(sigma) @ Vt
    """
    # Compute full SVD - use full_matrices=False for efficiency
    U, sigma, Vt = torch.linalg.svd(A, full_matrices=False)
    
    # Truncate to desired number of components
    k = min(n_components, len(sigma), min(A.shape))
    return U[:, :k], sigma[:k], Vt[:k, :]

def compute_svd(A, n_components, use_full_svd=False, n_oversamples=10):
    """
    Unified SVD interface - choose between full and randomized SVD.
    
    Args:
        A: Input matrix (m x n)
        n_components: Number of components to compute
        use_full_svd: If True, use full SVD; if False, use randomized SVD
        n_oversamples: Additional samples for randomized SVD (ignored for full SVD)
        
    Returns:
        U, sigma, Vt: SVD components where A ≈ U @ diag(sigma) @ Vt
    """
    if use_full_svd:
        return full_svd(A, n_components)
    else:
        return randomized_svd(A, n_components, n_oversamples)

def svdlozo_perturb_parameters(model, zo_random_seed, step, scaling_factor=1, debug_output=False):
    """
    Perturb model parameters using SVD-guided low-rank optimization.
    Instead of random U and V, we compute SVD of random Z and use its components.
    """
    # Only print tau on first step and when debug_output is True
    if step == 0 and debug_output:
        svd_method = "Full" if use_full_svd else "Randomized"
        print(f"[SVD-LOZO] Using τ={svd_tau}, max_rank={svd_max_rank}, SVD={svd_method}")
    
    torch.manual_seed(zo_random_seed)
    
    # Get parameters to optimize
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Handle _orig_mod prefix from torch.compile
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))
    
    # Debug info for first few steps only when requested
    rank_stats = []
    if step <= 2 and debug_output:
        print(f"\nSVD-LOZO Debug Info (step {step}):")
    
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # For matrices, use SVD-guided low-rank perturbation
            # Generate random matrix same size as parameter
            Z = torch.randn_like(param)
            
            # Use unified SVD interface
            U, sigma, Vt = compute_svd(Z, n_components=svd_max_rank, use_full_svd=use_full_svd)
            
            # Adaptive rank selection using τ threshold
            r_adaptive = (sigma > svd_tau * sigma[0]).sum().item()
            r_adaptive = max(r_adaptive, 1)  # At least rank 1
            
            # Collect rank statistics for summary
            if step <= 2 and debug_output:
                rank_stats.append({
                    'name': clean_name,
                    'rank': r_adaptive,
                    'sigma_max': sigma[0].item(),
                    'param_shape': param.shape
                })
            
            # Take the best r_adaptive components
            U_trunc = U[:, :r_adaptive]
            sigma_trunc = sigma[:r_adaptive]
            Vt_trunc = Vt[:r_adaptive, :]
            
            # Generate rank-1 perturbation using SVD directions (Option B)
            u_coeffs = torch.randn(r_adaptive, device=param.device, dtype=param.dtype)
            v_coeffs = torch.randn(r_adaptive, device=param.device, dtype=param.dtype)
            
            # Weight u_coeffs by singular values (Option 2)
            u_weighted = sigma_trunc * u_coeffs
            
            # Create perturbation directions
            u_direction = U_trunc @ u_weighted
            v_direction = Vt_trunc.T @ v_coeffs
            
            # Final rank-1 perturbation - use torch.outer for cleaner tensor ops
            perturbation = torch.outer(u_direction, v_direction)
            
            param.data = param.data + scaling_factor * perturbation * zo_eps
        else:
            # For vectors (biases), use Gaussian perturbation
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data = param.data + scaling_factor * z * zo_eps
    
    # Print rank statistics summary only when debug_output is True
    if step <= 2 and debug_output and len(rank_stats) > 0:
        avg_rank = sum(stat['rank'] for stat in rank_stats) / len(rank_stats)
        min_rank = min(stat['rank'] for stat in rank_stats)
        max_rank = max(stat['rank'] for stat in rank_stats)
        
        print(f"  Rank Statistics (τ={svd_tau}):")
        print(f"    Average rank: {avg_rank:.1f} / {svd_max_rank}")
        print(f"    Range: {min_rank} - {max_rank}")
        print(f"    Effective compression: {avg_rank/svd_max_rank*100:.1f}%")
        
        # Show a few examples of different layer types
        examples = {}
        for stat in rank_stats:
            layer_type = stat['name'].split('.')[-1] if '.' in stat['name'] else stat['name']
            if layer_type not in examples:
                examples[layer_type] = stat
        
        print(f"    Examples by layer type:")
        for layer_type, stat in examples.items():
            print(f"      {layer_type}: rank={stat['rank']}, σ_max={stat['sigma_max']:.1f}, shape={stat['param_shape']}")
    
    if step <= 2 and debug_output:
        print(f"Total parameters to optimize with SVD-LOZO: {len(named_parameters_to_optim)}")
    
    return named_parameters_to_optim

def svdlozo_step(model, X, Y, step, zo_random_seed):
    """
    Estimate gradient using SVD-LOZO with q-times finite differences.
    
    For q>1:
    1. Initialize gradient accumulator
    2. For each q iteration:
       - Generate SVD-guided perturbation
       - Compute (f+ - f-)/2*zo_eps  
       - Accumulate: g += (scalar coefficient) * SVD_perturbation
    3. Average: g = g / q
    """
    # If q=1, use the original implementation for maximum compatibility
    if zo_q == 1:
        # First function evaluation - only show debug on first call of each step
        named_params = svdlozo_perturb_parameters(model, zo_random_seed, step, scaling_factor=1, debug_output=True)
        loss1 = zo_forward(model, X, Y)

        # Second function evaluation - no debug output
        svdlozo_perturb_parameters(model, zo_random_seed, step, scaling_factor=-2, debug_output=False)
        loss2 = zo_forward(model, X, Y)

        # Calculate projected gradient
        projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()

        # Reset model back to its parameters at start of step - no debug output
        svdlozo_perturb_parameters(model, zo_random_seed, step, scaling_factor=1, debug_output=False)
        
        return loss1, projected_grad, named_params
    
    # For q>1, implement the multiple estimation approach
    # Initialize an empty gradient accumulator for each parameter
    accumulated_grads = {}
    first_loss = None
    
    # Create a list of parameters to optimize
    params_to_optimize = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            params_to_optimize.append((clean_name, param))
    
    # For each q estimation
    for q_idx in range(zo_q):
        # Sample a new random seed for this estimation
        current_zo_seed = np.random.randint(1000000000) if q_idx > 0 else zo_random_seed
        torch.manual_seed(current_zo_seed)
        
        # Generate SVD-guided perturbations for each parameter
        perturbations = {}  # Store the perturbations for this iteration
        
        for clean_name, param in params_to_optimize:
            if param.ndim >= 2:
                # For matrices, use SVD-guided low-rank perturbation
                Z = torch.randn_like(param)
                U, sigma, Vt = compute_svd(Z, n_components=svd_max_rank, use_full_svd=use_full_svd)
                
                # Adaptive rank selection using τ threshold
                r_adaptive = (sigma > svd_tau * sigma[0]).sum().item()
                r_adaptive = max(r_adaptive, 1)  # At least rank 1
                
                # Take the best r_adaptive components
                U_trunc = U[:, :r_adaptive]
                sigma_trunc = sigma[:r_adaptive] 
                Vt_trunc = Vt[:r_adaptive, :]
                
                # Generate rank-1 perturbation using SVD directions
                u_coeffs = torch.randn(r_adaptive, device=param.device, dtype=param.dtype)
                v_coeffs = torch.randn(r_adaptive, device=param.device, dtype=param.dtype)
                u_weighted = sigma_trunc * u_coeffs
                
                u_direction = U_trunc @ u_weighted
                v_direction = Vt_trunc.T @ v_coeffs
                perturbation = torch.outer(u_direction, v_direction)
                
                perturbations[clean_name] = perturbation
                # Apply positive perturbation
                param.data = param.data + perturbation * zo_eps
            else:
                # For vectors (biases), use Gaussian perturbation
                z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                perturbations[clean_name] = z
                # Apply positive perturbation
                param.data = param.data + z * zo_eps
        
        # Evaluate f(θ + perturbation)
        f_plus = zo_forward(model, X, Y)
        
        # Save first loss for reporting and debug (only on first q_idx and first step)
        if q_idx == 0:
            first_loss = f_plus
            # Show debug output only for first perturbation of first few steps
            if step <= 2:
                print(f"SVD-LOZO q={zo_q} estimation, step {step}")
        
        # Apply negative perturbation (from current state)
        for clean_name, param in params_to_optimize:
            if clean_name in perturbations:
                perturbation = perturbations[clean_name]
                param.data = param.data - 2 * perturbation * zo_eps
        
        # Evaluate f(θ - perturbation)
        f_minus = zo_forward(model, X, Y)
        
        # Compute scalar coefficient
        coeff = (f_plus - f_minus).item() / (2 * zo_eps)
        
        # Accumulate gradients for each parameter
        for clean_name, param in params_to_optimize:
            if clean_name in perturbations:
                perturbation = perturbations[clean_name]
                
                # Calculate gradient for this parameter
                grad = coeff * perturbation
                
                # Add to accumulated gradients
                if clean_name not in accumulated_grads:
                    accumulated_grads[clean_name] = grad
                else:
                    accumulated_grads[clean_name] += grad
        
        # Reset parameters to their original values
        for clean_name, param in params_to_optimize:
            if clean_name in perturbations:
                perturbation = perturbations[clean_name]
                param.data = param.data + perturbation * zo_eps
    
    # Average the gradients if q > 1
    if zo_q > 1:
        for clean_name in accumulated_grads:
            accumulated_grads[clean_name] = accumulated_grads[clean_name] / zo_q
    
    # Return the first loss and accumulated gradients  
    return first_loss, accumulated_grads, params_to_optimize

def svdlozo_update(model, optimizer, projected_grad, zo_random_seed, step, lr, named_parameters_to_optim=None):
    """
    Update model parameters using SVD-LOZO gradient estimate.
    """
    torch.manual_seed(zo_random_seed)
    
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Debug logging only on first step
    if step == 0 and master_process:
        print(f"Total parameters to optimize with SVD-LOZO: {len(named_parameters_to_optim)}")
    
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # Reproduce the same SVD-guided perturbation as in gradient estimation
            Z = torch.randn_like(param)
            U, sigma, Vt = compute_svd(Z, n_components=svd_max_rank, use_full_svd=use_full_svd)
            r_adaptive = (sigma > svd_tau * sigma[0]).sum().item()
            r_adaptive = max(r_adaptive, 1)
            
            U_trunc = U[:, :r_adaptive]
            sigma_trunc = sigma[:r_adaptive]
            Vt_trunc = Vt[:r_adaptive, :]
            
            u_coeffs = torch.randn(r_adaptive, device=param.device, dtype=param.dtype)
            v_coeffs = torch.randn(r_adaptive, device=param.device, dtype=param.dtype)
            u_weighted = sigma_trunc * u_coeffs
            
            u_direction = U_trunc @ u_weighted
            v_direction = Vt_trunc.T @ v_coeffs
            perturbation = torch.outer(u_direction, v_direction)
            
            # Apply update with weight decay
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (projected_grad * perturbation + weight_decay * param.data)
            else:
                param.data = param.data - lr * (projected_grad * perturbation)
        else:
            # For vectors, use Gaussian update
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (projected_grad * z + weight_decay * param.data)
            else:
                param.data = param.data - lr * (projected_grad * z)

# Function to update parameters using SVD-LOZO with direct gradient dictionary
def svdlozo_update_direct(model, optimizer, grad_dict, lr, named_parameters_to_optim=None):
    """
    Update model parameters using the pre-computed gradient dictionary from q-times SVD-LOZO estimation.
    """
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Apply the gradient update to each parameter
    for clean_name, param in named_parameters_to_optim:
        # Skip parameters not in gradient dictionary
        if clean_name not in grad_dict:
            continue
        
        # Get the gradient for this parameter
        grad = grad_dict[clean_name]
        
        # Apply update with or without weight decay
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        if is_weight:
            param.data = param.data - lr * (grad + weight_decay * param.data)
        else:
            param.data = param.data - lr * grad

# Function to perturb parameters with standard Gaussian perturbation (for MeZO)
def mezo_perturb_parameters(model, zo_random_seed, scaling_factor=1):
    """
    Perturb the parameters with standard Gaussian noise for MeZO.
    """
    torch.manual_seed(zo_random_seed)
    
    # Create a list of named parameters to optimize
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Handle _orig_mod prefix from torch.compile
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))
    
    for clean_name, param in named_parameters_to_optim:
        # Use standard Gaussian perturbation for all parameters
        z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
        param.data = param.data + scaling_factor * z * zo_eps
    
    return named_parameters_to_optim

# Function to estimate gradient using MeZO
def mezo_step(model, X, Y, step, zo_random_seed):
    """
    Estimate gradient using MeZO (Memory-efficient Zero-Order optimization).
    Return the loss from f(theta + z).
    """
    # First function evaluation
    named_params = mezo_perturb_parameters(model, zo_random_seed, scaling_factor=1)
    loss1 = zo_forward(model, X, Y)

    # Second function evaluation
    mezo_perturb_parameters(model, zo_random_seed, scaling_factor=-2)
    loss2 = zo_forward(model, X, Y)

    # Calculate projected gradient
    projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()

    # Reset model back to its parameters at start of step
    mezo_perturb_parameters(model, zo_random_seed, scaling_factor=1)
    
    return loss1, projected_grad, named_params

# Function to update parameters using MeZO
def mezo_update(model, optimizer, projected_grad, zo_random_seed, step, lr, named_parameters_to_optim=None):
    """
    Update model parameters using the MeZO gradient estimate.
    """
    torch.manual_seed(zo_random_seed)
    
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Debug logging only on first step
    if step == 0 and master_process:
        print(f"Total parameters to optimize with MeZO: {len(named_parameters_to_optim)}")
    
    for clean_name, param in named_parameters_to_optim:
        # Generate the same random perturbation as in the gradient estimation step
        z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
        
        # Standard weight decay handling
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        if is_weight:
            param.data = param.data - lr * (projected_grad * z + weight_decay * param.data)
        else:
            param.data = param.data - lr * (projected_grad * z)

# Function to update parameters using MeZO with momentum (MeZO-M)
def mezo_update_momentum(model, optimizer, projected_grad, zo_random_seed, exp_avg_m, step, lr, beta1=0.9, named_parameters_to_optim=None):
    """
    Update model parameters using the MeZO gradient estimate with momentum.
    
    Args:
        model: The model to update
        optimizer: The optimizer (not directly used but kept for API consistency)
        projected_grad: The projected gradient estimate from MeZO
        zo_random_seed: Random seed for reproducibility
        exp_avg_m: Dictionary of exponential moving average for momentum
        step: Current optimization step
        lr: Learning rate
        beta1: Momentum coefficient (default: 0.9)
        named_parameters_to_optim: List of (name, parameter) tuples to optimize
    """
    torch.manual_seed(zo_random_seed)
    
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Debug logs on first step
    if step == 0 and master_process:
        print(f"Total parameters to optimize with MeZO-M: {len(named_parameters_to_optim)}")
    
    for clean_name, param in named_parameters_to_optim:
        # Generate the same random perturbation as in the gradient estimation step
        z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
        
        # Initialize momentum if needed
        if clean_name not in exp_avg_m:
            exp_avg_m[clean_name] = torch.zeros_like(z)
        
        # Update momentum
        exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * projected_grad * z
        
        # Apply update with momentum
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        if is_weight:
            param.data = param.data - lr * (exp_avg_m[clean_name] + weight_decay * param.data)
        else:
            param.data = param.data - lr * exp_avg_m[clean_name]

# -----------------------------------------------------------------------------
# DiMeZO (Directional MeZO) specific functions
# -----------------------------------------------------------------------------

# Function to perturb parameters with directional selection (for DiMeZO)
def dimezo_perturb_parameters(model, zo_random_seed, scaling_factor=1, eps=None):
    """
    Perturb the parameters with standard Gaussian noise for DiMeZO directional selection.
    """
    if eps is None:
        eps = zo_eps  # Use global zo_eps if not provided
        
    torch.manual_seed(zo_random_seed)
    
    # Create a list of named parameters to optimize
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Handle _orig_mod prefix from torch.compile
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))
    
    for clean_name, param in named_parameters_to_optim:
        # Use standard Gaussian perturbation for all parameters
        z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
        param.data = param.data + scaling_factor * z * eps
    
    return named_parameters_to_optim

# Function to estimate gradient using DiMeZO with directional selection
def dimezo_step(model, X, Y, step, zo_random_seed, directional_q=None, eps=None, direct_movement=False):
    """
    Estimate gradient using DiMeZO (Directional MeZO) with directional selection.
    
    Algorithm:
    1. Compute baseline loss: f(θ)
    2. Sample q random directions z_1, ..., z_q
    3. Evaluate f(θ + ε*z_i) for each direction i
    4. Select z_best = argmin_zi f(θ + ε*z_i) (direction that gives lowest loss)
    5. Success = min_loss < baseline_loss
    6. If direct_movement=True: return Z_best directly for θ ← θ + α * Z_best
       Else: Estimate directional derivative: c = [f(θ + ε*z_best) - f(θ - ε*z_best)]/(2ε)
             This gives the projected gradient ∇f(θ) · z_best along z_best direction.
             Since z_best decreases loss, c < 0, so θ ← θ - α * c * z_best moves toward z_best.
    
    Args:
        model: The model to optimize
        X, Y: Input batch 
        step: Current step (for debugging)
        zo_random_seed: Random seed for reproducibility
        directional_q: Number of directions to try. If None, uses global `directional_q`.
        eps: Perturbation size (uses global zo_eps if None)
        direct_movement: If True, return best direction directly instead of gradient estimate
    
    Returns:
        best_loss: Loss from the best direction f(θ + ε*z_best)
        gradient_or_direction: Either directional derivative c or best direction z_best
        best_seed: Random seed that generated the best direction
        success: Boolean indicating if we found a direction better than baseline
    """
    
    current_directional_q = directional_q if directional_q is not None else globals()['directional_q']
    
    if eps is None:
        eps = zo_eps  # Use global zo_eps if not provided
    
    # Step 1: Compute baseline loss f(θ)
    baseline_loss = zo_forward(model, X, Y)
    
    # Phase 1: Directional Selection - try q directions and find the best
    best_loss = float('inf')
    best_seed = None
    best_direction = None
    
    # Debug info for first few steps
    if step <= 2 and master_process:
        mode_str = "Direct Movement" if direct_movement else "Gradient Estimation"
        print(f"DiMeZO ({mode_str}): Baseline loss {baseline_loss:.4f}, trying q={current_directional_q} directions, eps={eps:.2e}")
    
    # Store the directions to retrieve the best one later
    directions = {}
    
    for i in range(current_directional_q):
        # Generate a unique random seed for this direction
        direction_seed = zo_random_seed + i  # Offset from base seed for reproducibility
        
        # Generate and store the direction
        torch.manual_seed(direction_seed)
        direction = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                direction[name] = z
        directions[direction_seed] = direction
        
        # Apply perturbation in this direction
        dimezo_perturb_parameters(model, direction_seed, scaling_factor=1, eps=eps)
        
        # Evaluate loss at θ + ε*z_i
        loss = zo_forward(model, X, Y)
        
        # Check if this is the best direction so far
        if loss < best_loss:
            best_loss = loss
            best_seed = direction_seed
            best_direction = directions[direction_seed]
        
        # Reset model back to original parameters
        dimezo_perturb_parameters(model, direction_seed, scaling_factor=-1, eps=eps)
    
    # Determine success: did we find a direction better than baseline?
    success = best_loss < baseline_loss
    
    if direct_movement:
        # Return the best direction directly
        gradient_or_direction = best_direction
    else:
        # Phase 2: Gradient Estimation along best direction
        # Apply perturbation in best direction
        dimezo_perturb_parameters(model, best_seed, scaling_factor=1, eps=eps)
        f_plus = best_loss  # We already computed this
        
        # Apply negative perturbation: θ - ε*z_best
        dimezo_perturb_parameters(model, best_seed, scaling_factor=-2, eps=eps)
        f_minus = zo_forward(model, X, Y)
        
        # Calculate projected gradient coefficient
        gradient_or_direction = ((f_plus - f_minus) / (2 * eps)).item()
        
        # Reset model back to original parameters
        dimezo_perturb_parameters(model, best_seed, scaling_factor=1, eps=eps)
    
    # Debug info for first few steps
    if step <= 2 and master_process:
        success_str = "SUCCESS" if success else "FAILURE"
        if direct_movement:
            print(f"DiMeZO: Best loss {best_loss:.4f} vs baseline {baseline_loss:.4f} -> {success_str} (returning direction)")
        else:
            print(f"DiMeZO: Best loss {best_loss:.4f} vs baseline {baseline_loss:.4f} -> {success_str} (grad_coeff {gradient_or_direction:.6f})")
    
    return best_loss, gradient_or_direction, best_seed, success

# Function to update parameters using DiMeZO
def dimezo_update(model, optimizer, gradient_or_direction, best_seed, step, lr, named_parameters_to_optim=None, direct_movement=False):
    """
    Update model parameters using DiMeZO.
    
    Two update modes:
    1. Direct Movement (direct_movement=True): 
       θ ← θ + α * Z_best (move directly toward the best direction)
       
    2. Gradient Descent (direct_movement=False):
       θ ← θ - α * c * Z_best where c = ∇f(θ) · Z_best (directional derivative)
       Since Z_best is selected to decrease loss, c < 0, so this becomes:
       θ ← θ + α * |c| * Z_best
       This is standard gradient descent with adaptive step size |c|.
    
    Args:
        model: The model to update
        optimizer: The optimizer (not directly used but kept for API consistency)
        gradient_or_direction: Either directional derivative c or direction dictionary Z_best
        best_seed: Random seed that generated the best direction
        step: Current optimization step
        lr: Learning rate
        named_parameters_to_optim: List of (name, parameter) tuples to optimize
        direct_movement: If True, gradient_or_direction is Z_best; if False, it's c
    """
    
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Debug logging only on first step
    if step == 0 and master_process:
        mode_str = "Direct Movement" if direct_movement else "Gradient Estimation"
        print(f"DiMeZO Update ({mode_str}): {len(named_parameters_to_optim)} parameters")
    
    if direct_movement:
        # Direct movement: X ← X + α * Z_best (move toward best direction)
        # Weight decay is still applied as: X ← X - α * weight_decay * X
        direction_dict = gradient_or_direction
        for clean_name, param in named_parameters_to_optim:
            if clean_name in direction_dict:
                z = direction_dict[clean_name]
                # Standard weight decay handling
                is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
                if is_weight:
                    param.data = param.data + lr * z - lr * weight_decay * param.data
                else:
                    param.data = param.data + lr * z
    else:
        # Gradient descent mode: X ← X - α * c * Z_best
        # where c = ∇f(θ) · Z_best is the directional derivative (projected gradient)
        # Since Z_best decreases loss, c < 0, so we move toward Z_best with step size α * |c|
        torch.manual_seed(best_seed)
        projected_grad = gradient_or_direction
        
        for clean_name, param in named_parameters_to_optim:
            # Generate the same random direction as the best direction found
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            
            # Standard weight decay handling
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (projected_grad * z + weight_decay * param.data)
            else:
                param.data = param.data - lr * (projected_grad * z)

# -----------------------------------------------------------------------------
# DiLoZO (Directional Low-Rank Zero-Order) specific functions
# -----------------------------------------------------------------------------

def dilozo_perturb_parameters(model, v_dict, u_seed, step, scaling_factor=1, eps=None, current_rank=None):
    """Perturb model parameters with U_seed V^T for DiLoZO.
    V matrices are handled similarly to LOZO (updated every step_interval).
    U matrix is generated based on u_seed.
    """ 
    if eps is None:
        eps = zo_eps # Use global zo_eps if not provided
    if current_rank is None:
        current_rank = rank_r # Use default rank if not specified
    if current_rank == 0: # Avoid issues with rank 0
        current_rank = 1

    torch.manual_seed(u_seed) # Ensure U is generated based on this specific seed

    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))

    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # For matrices, use low-rank perturbation U V^T
            need_new_v = (step % step_interval == 0 or
                          clean_name not in v_dict or
                          v_dict[clean_name].size(1) != current_rank) # Rank changed
            
            if need_new_v:
                # Ensure v has the correct dimensions, especially for the second dimension (rank)
                v = torch.randn(param.size(1), int(current_rank), device=param.device, dtype=param.dtype)
                v_dict[clean_name] = v
            else:
                v = v_dict[clean_name]
            
            # Ensure u has the correct dimensions
            u = torch.randn(param.size(0), int(current_rank), device=param.device, dtype=param.dtype)
            param.data = param.data + scaling_factor * (u @ v.t()) * eps
        else:
            # For vectors (biases), use Gaussian perturbation z
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data = param.data + scaling_factor * z * eps
    
    return named_parameters_to_optim

def dilozo_step(model, X, Y, v_dict, step, zo_random_seed, directional_q=None, eps=None, current_rank=None):
    """Estimate gradient using DiLoZO (Directional Low-Rank Zero-Order).
    
    Algorithm:
    1. Compute baseline loss: f(θ)
    2. Sample q random low-rank directions U_i V^T for i = 1, ..., q
    3. Evaluate f(θ + ε*U_i V^T) for each direction i
    4. Select U_best V^T = argmin_i f(θ + ε*U_i V^T) (direction that gives lowest loss)
    5. Success = min_loss < baseline_loss
    6. Estimate directional derivative: c = [f(θ + ε*U_best V^T) - f(θ - ε*U_best V^T)]/(2ε)
       This gives the projected gradient ∇f(θ) · (U_best V^T) along the best direction.
       Since U_best V^T decreases loss, c < 0, so θ ← θ - α * c * U_best V^T moves toward U_best V^T.

    Args:
        directional_q: Number of directions to try. If None, uses global `directional_q`.
    """
    current_directional_q = directional_q if directional_q is not None else globals()['directional_q']

    if eps is None:
        eps = zo_eps
    if current_rank is None:
        current_rank = rank_r
    if current_rank == 0: # Avoid issues with rank 0
        current_rank = 1


    baseline_loss = zo_forward(model, X, Y)
    best_loss_val = float('inf')
    best_u_seed = None

    if step <= 2 and master_process:
        print(f"DiLoZO: Baseline loss {baseline_loss:.4f}, trying q={current_directional_q} directions, eps={eps:.2e}, rank={current_rank}")

    for i in range(current_directional_q):
        direction_u_seed = zo_random_seed + i
        dilozo_perturb_parameters(model, v_dict, direction_u_seed, step, scaling_factor=1, eps=eps, current_rank=current_rank)
        current_loss = zo_forward(model, X, Y)
        if current_loss < best_loss_val:
            best_loss_val = current_loss
            best_u_seed = direction_u_seed
        # Reset parameters to original state before trying next direction
        dilozo_perturb_parameters(model, v_dict, direction_u_seed, step, scaling_factor=-1, eps=eps, current_rank=current_rank)

    success = best_loss_val < baseline_loss

    # If no direction improved, best_u_seed might be None. Handle this.
    if best_u_seed is None: # This can happen if all directions yield worse or equal loss
        if master_process:
            print(f"DiLoZO Step: No improving direction found out of {current_directional_q}. Using first direction's seed for grad_coeff calculation.")
        best_u_seed = zo_random_seed # Fallback to the first seed

    # Parameter perturbation for f_plus
    dilozo_perturb_parameters(model, v_dict, best_u_seed, step, scaling_factor=1, eps=eps, current_rank=current_rank)
    # f_plus is ideally best_loss_val if the best_u_seed led to it, otherwise re-evaluate if necessary.
    # If best_u_seed was a fallback, best_loss_val might not correspond to f(theta + eps U_best V^T)
    # Re-evaluating f_plus ensures correctness.
    f_plus = zo_forward(model, X, Y) 
    
    # Parameter perturbation for f_minus
    # We perturbed by +1*eps*UVT to get f_plus. Now perturb by -2*eps*UVT from current state to get to (theta - eps*UVT)
    dilozo_perturb_parameters(model, v_dict, best_u_seed, step, scaling_factor=-2, eps=eps, current_rank=current_rank)
    f_minus = zo_forward(model, X, Y)
    
    grad_coeff = ((f_plus - f_minus) / (2 * eps)).item()
    
    # Reset parameters to original state (theta)
    # We are currently at (theta - eps*UVT). Add back 1*eps*UVT to get to theta
    dilozo_perturb_parameters(model, v_dict, best_u_seed, step, scaling_factor=1, eps=eps, current_rank=current_rank)

    if step <= 2 and master_process:
        success_str = "SUCCESS" if success else "FAILURE"
        print(f"DiLoZO: f+ {f_plus:.4f}, f- {f_minus:.4f}. Best loss {best_loss_val:.4f} vs baseline {baseline_loss:.4f} -> {success_str} (grad_coeff {grad_coeff:.6f})")

    return best_loss_val, grad_coeff, best_u_seed, success

def dilozo_update(model, optimizer, grad_coeff, best_u_seed, v_dict, step, lr, named_parameters_to_optim=None, current_rank=None):
    """Update model parameters using DiLoZO.
    
    Update rule: θ ← θ - α * (grad_coeff / r) * U_best V_best^T
    
    Mathematical reasoning:
    - grad_coeff = ∇f(θ) · (U_best V_best^T) is the directional derivative
    - Since U_best V_best^T was selected to decrease loss, grad_coeff < 0
    - The update θ ← θ - α * (negative) * U_best V_best^T moves toward U_best V_best^T
    - Division by rank r provides scaling for the low-rank structure
    """
    if current_rank is None:
        current_rank = rank_r
    if current_rank == 0:
        if master_process: print("Warning: DiLoZO update called with current_rank=0. Skipping update.")
        return

    torch.manual_seed(best_u_seed) # Regenerate U_best

    if named_parameters_to_optim is None:
        # This list comprehension was rebuilt from context.
        named_parameters_to_optim = [(name if not name.startswith('_orig_mod.') else name[len('_orig_mod.'):], param)
                                     for name, param in model.named_parameters() if param.requires_grad]


    if step <= 2 and master_process:
        print(f"DiLoZO Update: {len(named_parameters_to_optim)} parameters, rank={current_rank}, lr={lr:.2e}")

    effective_lr = lr / current_rank

    for clean_name, param in named_parameters_to_optim:
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        
        original_dtype = param.data.dtype # Store original dtype
        param_data_float32 = param.data.float() # Convert to float32 for update precision

        if param.ndim >= 2:
            u_best = torch.randn(param.size(0), int(current_rank), device=param.device, dtype=torch.float32) # Use float32 for U
            v_best = v_dict[clean_name].float() # Convert V to float32
            
            # Perform update calculation in float32
            update_term = effective_lr * grad_coeff * (u_best @ v_best.t())
            
            if is_weight and weight_decay > 0:
                param_data_float32 = param_data_float32 - (update_term + weight_decay * lr * param_data_float32)
            else:
                param_data_float32 = param_data_float32 - update_term
        else:
            # For vectors (biases)
            z_best = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=torch.float32) # Use float32 for Z
            update_term_vec = lr * grad_coeff * z_best
            
            if is_weight and weight_decay > 0:
                param_data_float32 = param_data_float32 - (update_term_vec + weight_decay * lr * param_data_float32)
            else:
                param_data_float32 = param_data_float32 - update_term_vec
        
        param.data = param_data_float32.to(original_dtype) # Convert back to original dtype


def dilozo_update_momentum(model, optimizer, grad_coeff, best_u_seed, v_dict, exp_avg_m, step, lr, beta1=0.9, named_parameters_to_optim=None, current_rank=None):
    """Update model parameters using DiLoZO with momentum.
    
    Momentum update:
    m_t = β1 * m_{t-1} + (1 - β1) * g_t
    θ_t = θ_{t-1} - α * m_t
    
    where g_t = (grad_coeff / r) * U_best V_best^T for matrices, and grad_coeff * Z_best for vectors
    
    Mathematical reasoning:
    - grad_coeff = ∇f(θ) · (U_best V_best^T) is the directional derivative
    - Since U_best V_best^T was selected to decrease loss, grad_coeff < 0
    - The momentum accumulates these negative gradients, creating consistent movement toward good directions
    """
    if current_rank is None:
        current_rank = rank_r
    if current_rank == 0:
        if master_process: print("Warning: DiLoZO Momentum update called with current_rank=0. Skipping update.")
        return

    torch.manual_seed(best_u_seed) # Regenerate U_best / Z_best

    if named_parameters_to_optim is None:
        named_parameters_to_optim = [(name if not name.startswith('_orig_mod.') else name[len('_orig_mod.'):], param)
                                     for name, param in model.named_parameters() if param.requires_grad]

    if step <= 2 and master_process:
        print(f"DiLoZO Momentum Update: {len(named_parameters_to_optim)} parameters, rank={current_rank}, lr={lr:.2e}, beta1={beta1}")

    # For matrix parameters, g_t is scaled by 1/rank. For vector parameters, effectively rank=1.
    # grad_coeff is (f+ - f-)/(2eps)
    
    for clean_name, param in named_parameters_to_optim:
        is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
        
        original_dtype = param.data.dtype
        param_data_float32 = param.data.float() # Convert param data to float32

        if param.ndim >= 2:
            u_best = torch.randn(param.size(0), int(current_rank), device=param.device, dtype=torch.float32) # U in float32
            v_best = v_dict[clean_name].float() # V in float32
            # g_t = (grad_coeff / current_rank) * (U_best @ V_best^T)
            g_t = (grad_coeff / current_rank) * (u_best @ v_best.t()) 
        else:
            # For vectors (biases), effectively rank is 1.
            z_best = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=torch.float32) # Z in float32
            # g_t = grad_coeff * Z_best
            g_t = grad_coeff * z_best

        if clean_name not in exp_avg_m:
            exp_avg_m[clean_name] = torch.zeros_like(param_data_float32) # Initialize momentum in float32
        
        current_momentum = exp_avg_m[clean_name].float() # Ensure momentum is float32
        current_momentum.mul_(beta1).add_(g_t, alpha=1 - beta1)
        exp_avg_m[clean_name] = current_momentum
        
        update_val = lr * current_momentum # This is m_t * lr

        if is_weight and weight_decay > 0:
            param_data_float32 = param_data_float32 - (update_val + weight_decay * lr * param_data_float32)
        else:
            param_data_float32 = param_data_float32 - update_val
            
        param.data = param_data_float32.to(original_dtype)

# -----------------------------------------------------------------------------
# KronZO (Kronecker Zero-Order) specific functions
# -----------------------------------------------------------------------------

def find_closest_divisor(n, target):
    """Find divisor of n closest to target"""
    divisors = [i for i in range(1, n+1) if n % i == 0]
    return min(divisors, key=lambda x: abs(x - target))

def choose_kron_dims_approx_square(d_out, d_in):
    """Choose Kronecker factorization dimensions close to square roots"""
    import math
    
    # Find factors closest to square roots
    target_out = int(math.sqrt(d_out))
    target_in = int(math.sqrt(d_in))
    
    # Find divisors closest to targets
    m1 = find_closest_divisor(d_out, target_out)
    n1 = d_out // m1
    
    m2 = find_closest_divisor(d_in, target_in)
    n2 = d_in // m2
    
    return m1, n1, m2, n2

def choose_kron_dims_fixed_factor(d_out, d_in, max_factor=32):
    """Choose Kronecker factorization with factors ≤ max_factor"""
    # For d_out
    out_factors = [i for i in range(1, min(max_factor+1, d_out+1)) if d_out % i == 0]
    m1 = max(out_factors) if out_factors else 1  # Largest factor ≤ max_factor
    n1 = d_out // m1
    
    # For d_in  
    in_factors = [i for i in range(1, min(max_factor+1, d_in+1)) if d_in % i == 0]
    m2 = max(in_factors) if in_factors else 1
    n2 = d_in // m2
    
    return m1, n1, m2, n2

def choose_kron_dims_power2(d_out, d_in):
    """Choose Kronecker factorization using largest power-of-2 divisors"""
    def largest_power2_divisor(n):
        if n == 0:
            return 1
        power = 0
        while n % (2 ** (power + 1)) == 0:
            power += 1
        return 2 ** power
    
    m1 = largest_power2_divisor(d_out)
    n1 = d_out // m1
    
    m2 = largest_power2_divisor(d_in)
    n2 = d_in // m2
    
    return m1, n1, m2, n2

def choose_kron_dims(d_out, d_in, strategy='approx_square', max_factor=32):
    """Choose Kronecker factorization dimensions based on strategy"""
    if strategy == 'approx_square':
        return choose_kron_dims_approx_square(d_out, d_in)
    elif strategy == 'fixed_factor':
        return choose_kron_dims_fixed_factor(d_out, d_in, max_factor)
    elif strategy == 'power2':
        return choose_kron_dims_power2(d_out, d_in)
    else:
        raise ValueError(f"Unknown Kronecker strategy: {strategy}")

def kronzo_perturb_parameters(model, zo_random_seed, step, scaling_factor=1, eps=None, strategy='approx_square', max_factor=32):
    """
    Perturb model parameters using Kronecker product structure A ⊗ B.
    For each matrix parameter W ∈ ℝ^(d_out × d_in), we:
    1. Choose factorization dimensions: m1×n1 = d_out, m2×n2 = d_in
    2. Sample A ∈ ℝ^(m1×n1), B ∈ ℝ^(m2×n2)
    3. Compute perturbation: ΔW = A ⊗ B
    """
    if eps is None:
        eps = zo_eps  # Use global zo_eps if not provided
        
    torch.manual_seed(zo_random_seed)
    
    # Create a list of named parameters to optimize
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Handle _orig_mod prefix from torch.compile
            clean_name = name
            if name.startswith('_orig_mod.'):
                clean_name = name[len('_orig_mod.'):]
            named_parameters_to_optim.append((clean_name, param))
    
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # For matrices, use Kronecker product perturbation
            d_out, d_in = param.shape
            
            # Choose Kronecker factorization dimensions
            m1, n1, m2, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
            
            # Sample A and B matrices
            A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
            B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
            
            # Compute Kronecker product A ⊗ B
            perturbation = torch.kron(A, B)
            
            # Apply perturbation
            param.data = param.data + scaling_factor * perturbation * eps
        else:
            # For vectors (biases), use Gaussian perturbation
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data = param.data + scaling_factor * z * eps
    
    return named_parameters_to_optim

def kronzo_step(model, X, Y, step, zo_random_seed, strategy='approx_square', max_factor=32, eps=None):
    """
    Estimate gradient using KronZO (Kronecker Zero-Order optimization).
    
    Algorithm:
    1. For each matrix parameter W ∈ ℝ^(d_out × d_in):
       - Choose factorization: m1×n1 = d_out, m2×n2 = d_in  
       - Sample A ∈ ℝ^(m1×n1), B ∈ ℝ^(m2×n2)
       - Perturbation: ΔW = A ⊗ B (Kronecker product)
    2. Compute finite difference: c = [f(θ + ε*ΔW) - f(θ - ε*ΔW)]/(2ε)
    3. This gives projected gradient ∇f(θ) · ΔW along Kronecker direction
    
    Args:
        model: The model to optimize
        X, Y: Input batch
        step: Current step (for debugging)
        zo_random_seed: Random seed for reproducibility
        strategy: Kronecker factorization strategy ('approx_square', 'fixed_factor', 'power2')
        max_factor: Maximum factor size for 'fixed_factor' strategy
        eps: Perturbation size (uses global zo_eps if None)
    
    Returns:
        loss: Loss from f(θ + ε*ΔW)
        projected_grad: Projected gradient coefficient c
        named_params: List of (name, parameter) tuples
    """
    if eps is None:
        eps = zo_eps  # Use global zo_eps if not provided
    
    # First function evaluation: f(θ + ε*ΔW)
    named_params = kronzo_perturb_parameters(model, zo_random_seed, step, scaling_factor=1, eps=eps, strategy=strategy, max_factor=max_factor)
    loss1 = zo_forward(model, X, Y)

    # Second function evaluation: f(θ - ε*ΔW)
    kronzo_perturb_parameters(model, zo_random_seed, step, scaling_factor=-2, eps=eps, strategy=strategy, max_factor=max_factor)
    loss2 = zo_forward(model, X, Y)

    # Calculate projected gradient: c = [f(θ + ε*ΔW) - f(θ - ε*ΔW)]/(2ε)
    projected_grad = ((loss1 - loss2) / (2 * eps)).item()

    # Reset model back to original parameters: θ
    kronzo_perturb_parameters(model, zo_random_seed, step, scaling_factor=1, eps=eps, strategy=strategy, max_factor=max_factor)
    
    return loss1, projected_grad, named_params

def kronzo_update(model, optimizer, projected_grad, zo_random_seed, step, lr, named_parameters_to_optim=None, strategy='approx_square', max_factor=32):
    """
    Update model parameters using KronZO gradient estimate.
    
    Update rule: θ ← θ - α * c * ΔW
    where ΔW = A ⊗ B is the Kronecker product perturbation
    and c is the projected gradient coefficient
    
    Args:
        model: The model to update
        optimizer: The optimizer (not directly used but kept for API consistency)
        projected_grad: The projected gradient coefficient c
        zo_random_seed: Random seed for reproducibility
        step: Current optimization step
        lr: Learning rate
        named_parameters_to_optim: List of (name, parameter) tuples to optimize
        strategy: Kronecker factorization strategy
        max_factor: Maximum factor size for 'fixed_factor' strategy
    """
    torch.manual_seed(zo_random_seed)
    
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Debug logging only on first step
    if step == 0 and master_process:
        print(f"KronZO Update: {len(named_parameters_to_optim)} parameters, strategy={strategy}, max_factor={max_factor}")
        
        # Show storage statistics for first few parameters
        total_original = 0
        total_kronecker = 0
        examples = []
        
        for i, (clean_name, param) in enumerate(named_parameters_to_optim[:5]):  # First 5 parameters
            if param.ndim >= 2:
                d_out, d_in = param.shape
                original_storage = d_out * d_in
                
                m1, n1, m2, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
                kronecker_storage = m1 * n1 + m2 * n2
                
                total_original += original_storage
                total_kronecker += kronecker_storage
                
                compression_ratio = kronecker_storage / original_storage
                examples.append({
                    'name': clean_name,
                    'shape': (d_out, d_in),
                    'factors': f"({m1}×{n1}) ⊗ ({m2}×{n2})",
                    'compression': f"{compression_ratio:.3f}"
                })
        
        if examples:
            print(f"KronZO Storage Examples:")
            for ex in examples:
                print(f"  {ex['name']}: {ex['shape']} → {ex['factors']}, compression: {ex['compression']}")
            
            overall_compression = total_kronecker / total_original if total_original > 0 else 0
            print(f"  Overall compression ratio: {overall_compression:.3f}")
    
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # For matrices, reproduce the same Kronecker perturbation
            d_out, d_in = param.shape
            
            # Choose the same factorization as in gradient estimation
            m1, n1, m2, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
            
            # Sample the same A and B matrices
            A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
            B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
            
            # Compute the same Kronecker product
            perturbation = torch.kron(A, B)
            
            # Apply update with weight decay
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (projected_grad * perturbation + weight_decay * param.data)
            else:
                param.data = param.data - lr * (projected_grad * perturbation)
        else:
            # For vectors, use Gaussian update
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (projected_grad * z + weight_decay * param.data)
            else:
                param.data = param.data - lr * (projected_grad * z)

def kronzo_update_momentum(model, optimizer, projected_grad, zo_random_seed, exp_avg_m, step, lr, beta1=0.9, named_parameters_to_optim=None, strategy='approx_square', max_factor=32):
    """
    Update model parameters using KronZO with momentum.
    
    Momentum update:
    m_t = β1 * m_{t-1} + (1 - β1) * g_t
    θ_t = θ_{t-1} - α * m_t
    
    where g_t = c * (A ⊗ B) for matrices, and c * Z for vectors
    
    Args:
        model: The model to update
        optimizer: The optimizer (not directly used but kept for API consistency)
        projected_grad: The projected gradient coefficient c
        zo_random_seed: Random seed for reproducibility
        exp_avg_m: Dictionary of exponential moving average for momentum
        step: Current optimization step
        lr: Learning rate
        beta1: Momentum coefficient (default: 0.9)
        named_parameters_to_optim: List of (name, parameter) tuples to optimize
        strategy: Kronecker factorization strategy
        max_factor: Maximum factor size for 'fixed_factor' strategy
    """
    torch.manual_seed(zo_random_seed)
    
    # If named_parameters_to_optim is not provided, create it
    if named_parameters_to_optim is None:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = name
                if name.startswith('_orig_mod.'):
                    clean_name = name[len('_orig_mod.'):]
                named_parameters_to_optim.append((clean_name, param))
    
    # Debug logs on first step
    if step == 0 and master_process:
        print(f"KronZO Momentum Update: {len(named_parameters_to_optim)} parameters, strategy={strategy}, beta1={beta1}")
    
    for clean_name, param in named_parameters_to_optim:
        if param.ndim >= 2:
            # For matrices, use Kronecker perturbation with momentum
            d_out, d_in = param.shape
            
            # Choose the same factorization as in gradient estimation
            m1, n1, m2, n2 = choose_kron_dims(d_out, d_in, strategy, max_factor)
            
            # Sample the same A and B matrices
            A = torch.randn(m1, n1, device=param.device, dtype=param.dtype)
            B = torch.randn(m2, n2, device=param.device, dtype=param.dtype)
            
            # Compute the same Kronecker product
            perturbation = torch.kron(A, B)
            
            # Initialize momentum if needed
            if clean_name not in exp_avg_m:
                exp_avg_m[clean_name] = torch.zeros_like(perturbation)
            
            # Update momentum: m_t = β1 * m_{t-1} + (1 - β1) * g_t
            exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * projected_grad * perturbation
            
            # Apply update with momentum
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (exp_avg_m[clean_name] + weight_decay * param.data)
            else:
                param.data = param.data - lr * exp_avg_m[clean_name]
        else:
            # For vectors, use Gaussian update with momentum
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            
            # Initialize or update momentum
            if clean_name not in exp_avg_m:
                exp_avg_m[clean_name] = projected_grad * z
            else:
                exp_avg_m[clean_name] = beta1 * exp_avg_m[clean_name] + (1 - beta1) * projected_grad * z
            
            # Apply update
            is_weight = "bias" not in clean_name and "layer_norm" not in clean_name and "layernorm" not in clean_name
            if is_weight:
                param.data = param.data - lr * (exp_avg_m[clean_name] + weight_decay * param.data)
            else:
                param.data = param.data - lr * exp_avg_m[clean_name]

# -----------------------------------------------------------------------------
# Adaptive zo_eps functions
# -----------------------------------------------------------------------------

class AdaptiveZoEps:
    """
    Adaptive zo_eps manager that adjusts perturbation size based on:
    1. Success/failure in finding good directions (DiMeZO-specific)
    2. Learning rate coupling (eps scales with lr decay)
    """
    
    def __init__(self, base_eps=1e-3, base_lr=6e-4, window_size=20, 
                 eps_increase_rate=0.05, eps_decrease_rate=0.1,
                 lr_coupling_strength=0.5, min_eps=1e-6, max_eps=1e-3):
        """
        Initialize adaptive zo_eps manager.
        
        Args:
            base_eps: Base perturbation size
            base_lr: Base learning rate for coupling
            window_size: Number of steps to track for success rate
            eps_increase_rate: Rate to increase eps when successful
            eps_decrease_rate: Rate to decrease eps when failing
            lr_coupling_strength: How strongly eps follows lr (0=no coupling, 1=full coupling)
            min_eps, max_eps: Bounds on eps values (1e-6 to 1e-3)
        """
        self.base_eps = base_eps
        self.base_lr = base_lr
        self.window_size = window_size
        self.eps_increase_rate = eps_increase_rate
        self.eps_decrease_rate = eps_decrease_rate
        self.lr_coupling_strength = lr_coupling_strength
        self.min_eps = min_eps
        self.max_eps = max_eps
        
        # State tracking
        self.current_eps = base_eps
        self.success_history = []
        self.step_count = 0
        
    def record_step(self, loss, projected_grad, learning_rate, method_info=None):
        """
        Record information from the current optimization step.
        
        Args:
            loss: Current loss value (not used, kept for compatibility)
            projected_grad: Projected gradient magnitude (not used, kept for compatibility)
            learning_rate: Current learning rate for coupling
            method_info: Dictionary with method-specific info, expects {'success': bool}
        """
        self.step_count += 1
        
        # Extract success from method_info
        if method_info and 'success' in method_info:
            is_successful = method_info['success']
        else:
            # Fallback: assume no success if info not provided
            is_successful = False
        
        # Update history with sliding window
        self.success_history.append(is_successful)
        
        # Keep only recent history
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)
        
        # Update eps immediately based on this single step
        self._update_eps_immediate(is_successful, learning_rate)
    
    def _update_eps_immediate(self, success, learning_rate):
        """Update eps immediately based on current step success and learning rate."""
        
        if success:
            # Success: increase exploration (but cap at max_eps)
            adaptive_factor = 1 + self.eps_increase_rate
        else:
            # Failure: decrease exploration (but cap at min_eps)
            adaptive_factor = 1 - self.eps_decrease_rate
        
        # Learning rate coupling: eps should scale with lr
        lr_factor = (learning_rate / self.base_lr) ** self.lr_coupling_strength
        
        # Apply both adjustments
        new_eps = self.current_eps * adaptive_factor * lr_factor
        
        # Apply bounds (1e-6 to 1e-3)
        self.current_eps = max(self.min_eps, min(self.max_eps, new_eps))
        
    def get_eps(self):
        """Get the current adaptive eps value."""
        return self.current_eps
    
    def get_stats(self):
        """Get statistics for logging/debugging."""
        if len(self.success_history) == 0:
            return {
                'eps': self.current_eps,
                'success_rate': 0.0,
                'steps_tracked': 0
            }
        
        return {
            'eps': self.current_eps,
            'success_rate': sum(self.success_history) / len(self.success_history),
            'steps_tracked': len(self.success_history)
        }

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
    v_dict = checkpoint.get('v_dict', {})  # Load LOZO state if available
    step = checkpoint.get('step', 0)  # Load step counter if available
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

# Initialize LOZO dictionary for low-rank vectors
if not init_from == 'resume':
    v_dict = {}
    step = 0
    exp_avg_m = {}  # For momentum (LOZO-M, MeZO-M, DiLoZO-M)
    v_old_dict = {} # For momentum with step_interval (LOZO-M)
elif 'exp_avg_m' in checkpoint and (use_momentum and (train_method == 'lozom' or train_method == 'mezom' or train_method == 'dilozo' or train_method == 'kronzo')):
    exp_avg_m = checkpoint['exp_avg_m']
    v_old_dict = checkpoint.get('v_old_dict', {}) # v_old_dict is specific to LOZO-M
else:
    exp_avg_m = {} # Initialize if not loaded
    v_old_dict = {} # Initialize if not loaded

# Initialize adaptive zo_eps manager
if use_adaptive_eps:
    adaptive_eps_manager = AdaptiveZoEps(
        base_eps=zo_eps,
        base_lr=learning_rate,
        window_size=adaptive_eps_window,
        eps_increase_rate=0.05,  # 5% increase on success
        eps_decrease_rate=0.1    # 10% decrease on failure
    )
    if master_process:
        print(f"Using adaptive zo_eps with base_eps={zo_eps}, range=[1e-6, 1e-3]")
else:
    adaptive_eps_manager = None

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
            wandb_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            
            # Add rank info for rank-adaptive LOZO
            if rank_adaptive and (train_method == 'lozo' or train_method == 'lozom' or train_method == 'dilozo'): # Added 'dilozo'
                current_rank = get_current_rank(iter_num, max_iters, min_rank, max_rank)
                wandb_dict["rank"] = current_rank
                wandb_dict["rank_progress"] = min(iter_num / max_iters, 1.0)
            
            wandb.log(wandb_dict)
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
                    'v_dict': v_dict,  # Save LOZO state
                    'step': step,      # Save step counter
                }
                if train_method == 'lozom' or train_method == 'mezom':
                    checkpoint['exp_avg_m'] = exp_avg_m
                    if train_method == 'lozom':
                        checkpoint['v_old_dict'] = v_old_dict
                elif train_method == 'kronzo' and use_momentum:
                    checkpoint['exp_avg_m'] = exp_avg_m
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward-backward update with different optimization methods
    if train_method == 'lozo' or train_method == 'lozom':
        # LOZO or LOZO-M optimization
        
        # Compute current rank for adaptive scheduling
        current_rank = get_current_rank(iter_num, max_iters, min_rank, max_rank)
        
        # Debug: Print rank info on first few iterations
        if iter_num <= 5 and master_process and rank_adaptive:
            progress = min(iter_num / max_iters, 1.0)
            print(f"Rank-Adaptive LOZO: iter {iter_num}, progress {progress:.3f}, rank {current_rank}/{max_rank}")
        
        for micro_step in range(gradient_accumulation_steps):
            # Create a random seed for this step
            current_zo_seed = np.random.randint(1000000000)
            
            # Estimate gradient with LOZO
            if zo_q == 1:
                # Use original implementation for q=1
                loss, projected_grad, _ = lowrank_zo_step(model, X, Y, v_dict, step, current_zo_seed, current_rank=current_rank)
                
                # Scale the loss to account for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Accumulate the projected gradient
                if micro_step == 0:
                    accumulated_grad = projected_grad / gradient_accumulation_steps
                else:  # Corrected indentation (was 8 spaces, now 20)
                    accumulated_grad += projected_grad / gradient_accumulation_steps
            else:  # Corrected indentation (was 12 spaces, now 16)
                # Use new q-times implementation
                loss, grad_dict, _ = lowrank_zo_step(model, X, Y, v_dict, step, current_zo_seed, current_rank=current_rank)
                
                # Scale the loss to account for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Scale and accumulate the gradients
                if micro_step == 0:
                    accumulated_grads = {}
                    for name, grad in grad_dict.items():
                        accumulated_grads[name] = grad / gradient_accumulation_steps
                else:
                    for name, grad in grad_dict.items():
                        accumulated_grads[name] += grad / gradient_accumulation_steps
            
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')

        # Update parameters using LOZO or LOZO-M
        if zo_q == 1:
            if train_method == 'lozom':
                lowrank_zo_update_momentum(raw_model, optimizer, accumulated_grad, current_zo_seed, v_dict, exp_avg_m, v_old_dict, step, lr, momentum_beta, current_rank=current_rank)
            else:
                lowrank_zo_update(raw_model, optimizer, accumulated_grad, current_zo_seed, v_dict, step, lr, current_rank=current_rank)
        else:
            # Use direct gradient update with accumulated gradients
            lowrank_zo_update_direct(raw_model, optimizer, accumulated_grads, lr)
        
        # Increment step counter after parameter update (not per microbatch)
        step += 1
    
    elif train_method == 'svdlozo':
        # SVD-LOZO optimization
        for micro_step in range(gradient_accumulation_steps):
            # Create a random seed for this step
            current_zo_seed = np.random.randint(1000000000)
            
            # Estimate gradient with SVD-LOZO
            if zo_q == 1:
                # Use original implementation for q=1 - only debug on first micro-step
                if micro_step == 0:
                    loss, projected_grad, named_params = svdlozo_step(model, X, Y, step, current_zo_seed)
                else:
                    # For subsequent micro-steps, we need a modified version that doesn't show debug
                    # First function evaluation - no debug output
                    svdlozo_perturb_parameters(model, current_zo_seed, step, scaling_factor=1, debug_output=False)
                    loss1 = zo_forward(model, X, Y)
                    # Second function evaluation - no debug output  
                    svdlozo_perturb_parameters(model, current_zo_seed, step, scaling_factor=-2, debug_output=False)
                    loss2 = zo_forward(model, X, Y)
                    # Calculate projected gradient
                    projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()
                    # Reset model back to its parameters at start of step - no debug output
                    svdlozo_perturb_parameters(model, current_zo_seed, step, scaling_factor=1, debug_output=False)
                    loss = loss1
                
                # Scale the loss to account for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Accumulate the projected gradient
                if micro_step == 0:
                    accumulated_grad = projected_grad / gradient_accumulation_steps
                else:
                    accumulated_grad += projected_grad / gradient_accumulation_steps
            else:
                # Use new q-times implementation
                loss, grad_dict, named_params = svdlozo_step(model, X, Y, step, current_zo_seed)
                
                # Scale the loss to account for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Scale and accumulate the gradients
                if micro_step == 0:
                    accumulated_grads = {}
                    for name, grad in grad_dict.items():
                        accumulated_grads[name] = grad / gradient_accumulation_steps
                else:
                    for name, grad in grad_dict.items():
                        accumulated_grads[name] += grad / gradient_accumulation_steps
            
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')

        # Update parameters using SVD-LOZO
        if zo_q == 1:
            # Use standard projected gradient update for q=1
            svdlozo_update(raw_model, optimizer, accumulated_grad, current_zo_seed, step, lr, named_params)
        else:
            # Use direct gradient update with accumulated gradients for q>1
            # We need a new function similar to lowrank_zo_update_direct but for SVD-LOZO
            svdlozo_update_direct(raw_model, optimizer, accumulated_grads, lr)
        
        # Increment step counter after parameter update (not per microbatch)
        step += 1
    
    elif train_method == 'mezo' or train_method == 'mezom':
        # MeZO or MeZO-M optimization
        for micro_step in range(gradient_accumulation_steps):
            # Create a random seed for this step
            current_zo_seed = np.random.randint(1000000000)
            
            # Estimate gradient with MeZO
            loss, projected_grad, named_params = mezo_step(model, X, Y, step, current_zo_seed)
            
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            
            # Scale the loss to account for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Accumulate the projected gradient
            if micro_step == 0:
                accumulated_grad = projected_grad / gradient_accumulation_steps
            else:
                accumulated_grad += projected_grad / gradient_accumulation_steps

        # Update parameters using MeZO or MeZO-M
        if train_method == 'mezom':
            mezo_update_momentum(raw_model, optimizer, accumulated_grad, current_zo_seed, exp_avg_m, step, lr, momentum_beta, named_params)
        else:
            mezo_update(raw_model, optimizer, accumulated_grad, current_zo_seed, step, lr, named_params)
        
        # Increment step counter after parameter update (not per microbatch)
        step += 1
    
    elif train_method == 'dimezo':
        # DiMeZO (Directional MeZO) optimization
        for micro_step in range(gradient_accumulation_steps):
            # Create a random seed for this step
            current_zo_seed = np.random.randint(1000000000)
            
            # Estimate gradient with DiMeZO directional selection
            loss, projected_grad, best_seed, success = dimezo_step(model, X, Y, step, current_zo_seed, directional_q=directional_q, eps=zo_eps, direct_movement=dimezo_direct_movement)
            
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            
            # Scale the loss to account for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Accumulate the projected gradient
            if micro_step == 0:
                accumulated_grad = projected_grad / gradient_accumulation_steps
                best_direction_seed = best_seed  # Store the best seed for update
                # Store first microbatch info for adaptive eps
                first_microbatch_loss = loss
                first_microbatch_success = success
            else:
                accumulated_grad += projected_grad / gradient_accumulation_steps

        # Update parameters using DiMeZO with the best direction from the first microbatch
        # Note: We use the first microbatch's best direction for consistency
        dimezo_update(raw_model, optimizer, accumulated_grad, best_direction_seed, step, lr, direct_movement=dimezo_direct_movement)
        
        # Record step information for adaptive eps (use first microbatch data)
        if use_adaptive_eps and adaptive_eps_manager is not None:
            # Use the success information from the first microbatch for adaptive eps
            adaptive_eps_manager.record_step(
                loss=first_microbatch_loss,  # Loss from first microbatch
                projected_grad=accumulated_grad,  # Accumulated gradient
                learning_rate=lr,
                method_info={'success': first_microbatch_success}  # Method info from first microbatch
            )
            
            # Update global zo_eps for next iteration
            zo_eps = adaptive_eps_manager.get_eps()
        
        # Increment step counter after parameter update (not per microbatch)
        step += 1
    
    elif train_method == 'dilozo':
        # DiLoZO (Directional LoZO) optimization
        current_rank = get_current_rank(iter_num, max_iters, min_rank, max_rank)

        # Debug: Print rank info on first few iterations
        if iter_num <= 5 and master_process and rank_adaptive:
            progress = min(iter_num / max_iters, 1.0)
            print(f"Rank-Adaptive DiLoZO: iter {iter_num}, progress {progress:.3f}, rank {current_rank}/{max_rank}")

        for micro_step in range(gradient_accumulation_steps):
            # Create a random seed for this step
            current_zo_seed = np.random.randint(1000000000)
            
            # Estimate gradient with DiLoZO directional selection
            # iter_loss is the unscaled loss from dilozo_step for this micro_iteration
            iter_loss, grad_coeff, best_u_seed, success = dilozo_step(model, X, Y, v_dict, step, current_zo_seed, directional_q=directional_q, eps=zo_eps, current_rank=current_rank)
            
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            
            # Scale the loss to account for gradient accumulation.
            # The 'loss' variable will be overwritten in each micro_step and hold the value
            # from the last micro_step, to be used by the logging section.
            loss = iter_loss / gradient_accumulation_steps
            
            # Accumulate the gradient coefficient
            if micro_step == 0:
                accumulated_grad_coeff = grad_coeff / gradient_accumulation_steps
                best_direction_u_seed = best_u_seed  # Store the best seed for update
                # Store first microbatch info for adaptive eps
                first_microbatch_loss = loss # Use the scaled loss from the first micro_batch
                first_microbatch_success = success
            else:
                accumulated_grad_coeff += grad_coeff / gradient_accumulation_steps

        # Update parameters using DiLoZO with the best direction from the first microbatch
        # Note: We use the first microbatch's best direction for consistency
        if use_momentum:
            dilozo_update_momentum(raw_model, optimizer, accumulated_grad_coeff, best_direction_u_seed, v_dict, exp_avg_m, step, lr, momentum_beta, current_rank=current_rank)
        else:
            dilozo_update(raw_model, optimizer, accumulated_grad_coeff, best_direction_u_seed, v_dict, step, lr, current_rank=current_rank)
        
        # Record step information for adaptive eps (use first microbatch data)
        if use_adaptive_eps and adaptive_eps_manager is not None:
            # Use the success information from the first microbatch for adaptive eps
            adaptive_eps_manager.record_step(
                loss=first_microbatch_loss,  # Loss from first microbatch
                projected_grad=accumulated_grad_coeff,  # Accumulated gradient coefficient
                learning_rate=lr,
                method_info={'success': first_microbatch_success}  # Method info from first microbatch
            )
            
            # Update global zo_eps for next iteration
            zo_eps = adaptive_eps_manager.get_eps()
        
        # Increment step counter after parameter update (not per microbatch)
        step += 1
    
    elif train_method == 'kronzo':
        # KronZO (Kronecker Zero-Order) optimization
        for micro_step in range(gradient_accumulation_steps):
            # Create a random seed for this step
            current_zo_seed = np.random.randint(1000000000)
            
            # Estimate gradient with KronZO
            loss, projected_grad, named_params = kronzo_step(model, X, Y, step, current_zo_seed, strategy=kron_strategy, max_factor=kron_max_factor, eps=zo_eps)
            
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            
            # Scale the loss to account for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Accumulate the projected gradient
            if micro_step == 0:
                accumulated_grad = projected_grad / gradient_accumulation_steps
            else:
                accumulated_grad += projected_grad / gradient_accumulation_steps

        # Update parameters using KronZO or KronZO with momentum
        if use_momentum:
            kronzo_update_momentum(raw_model, optimizer, accumulated_grad, current_zo_seed, exp_avg_m, step, lr, momentum_beta, named_params, strategy=kron_strategy, max_factor=kron_max_factor)
        else:
            kronzo_update(raw_model, optimizer, accumulated_grad, current_zo_seed, step, lr, named_params, strategy=kron_strategy, max_factor=kron_max_factor)
        
        # Increment step counter after parameter update (not per microbatch)
        step += 1
    
    else:
        # Traditional optimizer (Adam or SGD)
        model.train()
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # scale the loss
            loss.backward()
            
            # Immediately async prefetch next batch while model is doing backward pass on the GPU
            X, Y = get_batch('train')
        
        # Clip gradients
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        # Only print progress on master process to avoid log pollution
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
        
        # Get adaptive eps stats for logging
        eps_info = {}
        if use_adaptive_eps and adaptive_eps_manager is not None:
            eps_stats = adaptive_eps_manager.get_stats()
            eps_info = {
                'eps': f"{eps_stats['eps']:.2e}",
                'success': f"{eps_stats['success_rate']:.2f}"
            }
        
        # Update progress bar
        pbar_info = {
            'loss': f'{lossf:.4f}',
            'lr': f'{lr:.2e}',
            'mem': f'{memory_allocated:.1f}GB',
            'mfu': f'{running_mfu*100:.1f}%',
            'step': step
        }
        
        # Add rank info for rank-adaptive LOZO
        if rank_adaptive and (train_method == 'lozo' or train_method == 'lozom' or train_method == 'dilozo'): # Added 'dilozo'
            current_rank = get_current_rank(iter_num, max_iters, min_rank, max_rank)
            pbar_info['rank'] = f'{current_rank}/{max_rank}'
        
        # Add adaptive eps info to progress bar if enabled
        if eps_info:
            pbar_info.update(eps_info)
        
        pbar.set_postfix(pbar_info)
        pbar.update(1)
    
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

pbar.close()

if ddp:
    destroy_process_group() 