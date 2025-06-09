import torch
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from train import TrainerConfig
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load('out-calls/ckpt.pt', map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

with open('data/calls/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

while True:
    user_input = input("You (Caller): ")
    prompt = f"Caller: {user_input}\nOperator:"
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model.generate(context, max_new_tokens=100)
    response = decode(out[0].tolist())
    print(response[len(prompt):].strip())
