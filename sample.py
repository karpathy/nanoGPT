"""
Sample from a trained model
"""
import os
import torch
import tiktoken
from model import GPTConfig, GPT

device = 'cuda:2'
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

out_dir = 'out'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)

# model
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
#start = enc.encode("\n")
start = [enc.eot_token]
x = (torch.tensor(start, dtype=torch.long, device=device)[None, ...])

for k in range(1):

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = model.generate(x, 300, temperature=0.8, top_k=200)

    print(enc.decode(y[0].tolist()))
    print('---------------')
