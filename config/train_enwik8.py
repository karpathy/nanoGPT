# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out'
eval_interval = 500
eval_iters = 50
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'enwik8'
wandb_run_name = 'mini-gpt'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 2
n_embd = 512
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 15000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
init_from = "scratch"

compile = True
# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
concrete_dropout = True
N = 100_000_000 # number of tokens in the dataset
l = 2 # prior lengthscale -> expected magnitude of the weights (1 as default). Higher -> larger weights
weight_tau = .01 # model precision. Common default is 1. Different choice is 1/sigma^2 (sigma^2 is variance of residuals)
weight_reg_weight = l**2 / (weight_tau * N)
dropout_reg_weight = 2/(weight_tau * N)
assert weight_reg_weight/dropout_reg_weight == l**2/2
print(f"weight_reg_weight: {weight_reg_weight}")
print(f"dropout_reg_weight: {dropout_reg_weight}")
