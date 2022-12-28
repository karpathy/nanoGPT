"""
A much shorter version of train.py for benchmarking the model
"""

import time
import torch
from model import GPTConfig, GPT

device = 'cuda:3'
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.manual_seed(1337)

batch_size = 8
block_size = 1024

gptconf = GPTConfig(
    block_size = block_size, # how far back does the model look? i.e. context size
    n_layer = 12, n_head = 12, n_embd = 768, # size of the model
    dropout = 0, # for determinism
)
model = GPT(gptconf)
model.to(device)

x = torch.randint(50257, (batch_size, block_size), device=device)
y = torch.randint(50257, (batch_size, block_size), device=device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95))

burn_in = 10 # number of burn in steps where we don't measure time
num_steps = 30
for k in range(num_steps):

    if k == burn_in:
        t0 = time.time() # start the timer

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    lossf = loss.item()
    print(f"{k}/{num_steps} loss: {lossf:.4f}")

torch.cuda.synchronize()
t1 = time.time()
print("time in ms per iteration: %.2f" % ((t1 - t0) / (num_steps - burn_in) * 1000))
