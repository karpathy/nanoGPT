out_dir = 'out-transcendence-gpt-30-train-1-test'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 1000
log_interval = 10 # don't print too too often

always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'transcendence-gpt'
wandb_run_name = 'first_data_30_train_1_test'

dataset = 'card_set_30_train_1_test'
gradient_accumulation_steps = 4
batch_size = 16
block_size = 540 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0

learning_rate = 3e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.90 # make a bit bigger because number of tokens per iter is small

# warmup_iters = 100 # not super necessary potentially

vocab_size = 17

# # on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

dtype = "float32"