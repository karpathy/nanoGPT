import time

out_dir = 'out-wikitext'
eval_interval = 5
log_interval = 1
eval_iters = 5
wandb_log = False # feel free to turn on
wandb_project = 'wikitext'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'wikitext'
init_from = 'gpt2'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# make the run super light for testing
batch_size = 1
gradient_accumulation_steps = 1
max_iters = 10
block_size = 64

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
compile = False
