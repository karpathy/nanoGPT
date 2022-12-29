"""
A much shorter version of train.py for benchmarking
"""
import os
import numpy as np
import time
import torch
from model import GPTConfig, GPT

device = 'cuda:3'
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.manual_seed(1337)

batch_size = 8
block_size = 1024

# data loading init
real_data = True
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # note ignore split in benchmarking script
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(50257, (batch_size, block_size), device=device)
    y = torch.randint(50257, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

# model init
gptconf = GPTConfig(
    block_size = block_size, # how far back does the model look? i.e. context size
    n_layer = 12, n_head = 12, n_embd = 768, # size of the model
    dropout = 0, # for determinism
)
model = GPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95))

burn_in = 10 # number of burn in steps where we don't measure time
num_steps = 30
for k in range(num_steps):

    if k == burn_in:
        t0 = time.time() # start the timer

    X, Y = get_batch('train')
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(X, Y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    lossf = loss.item()
    print(f"{k}/{num_steps} loss: {lossf:.4f}")

torch.cuda.synchronize()
t1 = time.time()
print("time in ms per iteration: %.2f" % ((t1 - t0) / (num_steps - burn_in) * 1000))
