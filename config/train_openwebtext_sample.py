# Real (non-toy) OpenWebText sample run for single-GPU loss curves.
#
# Dataset must be prepared first:
#   python data/openwebtext_sample/prepare.py --subset 'train[:1%]' --val_frac 0.01

wandb_log = False
dataset = "openwebtext_sample"

# keep tokens/iter reasonable for 1× A100-80GB
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 4  # tokens/iter = 8*1024*4 = 32,768

# training length (adjust as needed)
max_iters = 5000
lr_decay_iters = 5000

# log + eval for curves
log_interval = 1
eval_interval = 200
eval_iters = 50

# reproducibility / simplicity
compile = False

