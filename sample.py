"""
Sample from a trained model
"""
import os
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out'
device = 'cuda'
compile = False
start = "\n" # or "<|endoftext|>" or whatever you like
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # higher temperature (up to 1) is more random, lower (down to 0) means more greedy
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# encode the beginning of the prompt
enc = tiktoken.get_encoding("gpt2")
start_ids = enc.encode(start, allowed_special={"<|endoftext|>"})
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

for k in range(num_samples):

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    print(enc.decode(y[0].tolist()))
    print('---------------')
