# Real OpenWebText sample run at longer context (4k) for single-GPU loss curves.
#
# Dataset must be prepared first:
#   python data/openwebtext_sample/prepare.py --subset 'train[:1%]' --val_frac 0.01

wandb_log = False
dataset = "openwebtext_sample"

# Keep tokens/iter constant vs the 1k config:
#   2 * 4096 * 4 = 32,768 tokens/iter
batch_size = 2
block_size = 4096
gradient_accumulation_steps = 4

max_iters = 5000
lr_decay_iters = 5000

log_interval = 1
eval_interval = 200
eval_iters = 50

compile = False

