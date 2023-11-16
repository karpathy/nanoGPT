# Custom Config for the Perplexity blog Work


# I/O
log_interval = 10
eval_interval = 200
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
out_dir = 'insulted'  # we can override this fellow with a cli arg 

# wandb logging
# TODO@ckg: set up for logging training runs?
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2 - decayed LR - ' + str(time.time())

# data
dataset = 'shakespeare_insults'
gradient_accumulation_steps = 5 * 4 # used to simulate larger batch sizes
batch_size = 42 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 64 # basically the context window in the context of this repo
vocab_size = 328

# model
n_layer = 12
n_head = 16
n_embd = 1024
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate = 3e-4 # max learning rate
max_iters = 6000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 6000 # should be ~= max_iters per Chinchilla
min_lr = 3e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
compile = True # use PyTorch 2.0 to compile the model to be faster

