import time
from functools import partial

import torch
from minlora import LoRAParametrization

out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-xl' # this is the largest GPT-2 model
init_from = 'gpt2-large' # use a smaller for faster training
# xl doesn't fit on 24GB GPU, but with LORA it does

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False


use_lora = True
learning_rate = 1e-3 # use a higher LR for LoRA
lora_dropout_p = 0.0
rank=4
lora_alpha = 64
lora_config = {
    torch.nn.Embedding: {
        "weight": partial(LoRAParametrization.from_embedding, rank=rank, lora_alpha=lora_alpha),
    },
    torch.nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=lora_alpha),
    },
}