# train a miniature character-level shakespeare model with Attention Residuals
# compare with train_shakespeare_char.py to see the effect of AttnRes

out_dir = 'out-shakespeare-char-attnres'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-attnres'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# baby GPT model with Attention Residuals
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
attn_res = True
attn_res_block_size = 4  # 2 transformer layers per block

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# AttnRes uses dynamic list ops that may cause torch.compile recompilation;
# set compile = False if you encounter issues
# compile = False
