import time

out_dir = 'out-shakespeare'
eval_interval = 200
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())
compile = False # takes too little time to finetune, not worth it

# save a nice and overfit checkpoint that
# will only speak Shakespeare and forgets
# everything else about the world #dark
always_save_checkpoint = True

dataset = 'shakespeare'
init_from = 'gpt2-xl'
batch_size = 1
block_size = 512

learning_rate = 1e-5
max_iters = 1000
decay_lr = False
