# train a State Space Model on character-level shakespeare
# demonstrates SSM architecture for language modeling

out_dir = 'out-ssm-shakespeare-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True
wandb_project = 'synthetic-lm'
wandb_run_name = 'ssm-shakespeare-char'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# SSM model configuration
model_type = 'ssm'  # Use State Space Model instead of GPT
n_layer = 6
n_head = 6  # not used in SSM, but kept for compatibility
n_embd = 384
dropout = 0.2

# SSM-specific hyperparameters
ssm_state_dim = 16   # dimension of SSM hidden state
ssm_conv_dim = 4     # convolution kernel size for local context
ssm_expand = 2       # expansion factor (d_inner = expand * n_embd)

# optimizer settings
learning_rate = 1e-3  # with small models can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
