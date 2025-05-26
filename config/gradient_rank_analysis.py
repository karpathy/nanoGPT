# Configuration for Gradient Rank Analysis
# Analyzes the effective rank of gradient matrices during SGD training

# Output directory
out_dir = 'out-gradient-rank-analysis'

# Logging settings
eval_interval = 200  # Evaluate less frequently to focus on gradient analysis
log_interval = 25
eval_iters = 25
eval_only = False
always_save_checkpoint = False  # Don't need frequent checkpoints for analysis

# Training data
dataset = 'shakespeare'
gradient_accumulation_steps = 1  # Keep simple for analysis
batch_size = 32
block_size = 256  # Smaller context for faster training

# Model configuration (small GPT for quick analysis)
n_layer = 6
n_head = 6
n_embd = 384  # embedding dimension
dropout = 0.1
bias = False

# SGD optimizer settings
learning_rate = 1e-3
max_iters = 1000  # Shorter run for analysis
weight_decay = 1e-1
beta1 = 0.9  # Not used by SGD but needed for compatibility
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 100
lr_decay_iters = 1000
min_lr = 1e-4

# Training method (force SGD for gradient analysis)
train_method = 'sgd'

# Gradient rank analysis specific parameters
rank_analysis_interval = 20  # Analyze every 20 steps for good temporal resolution
svd_tau_thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]  # Different thresholds for effective rank
save_singular_values = True  # Save full distributions for detailed analysis
max_rank_to_analyze = 200  # Analyze up to rank 200 (should cover most layers)

# System settings
device = 'cuda'
dtype = 'bfloat16'
compile = True 