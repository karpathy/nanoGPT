"""
Sample from a trained model (Modified for AIML332 Assignment 2, Task 1.1)
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import matplotlib.pyplot as plt # Added for plotting
import torch.nn.functional as F # Added for softmax

# -----------------------------------------------------------------------------
# Task 1.1: Add a new command-line argument for showing probabilities
# We add it here directly. In your setup, you might add this to 'configurator.py'
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--init_from', type=str, default='gpt2', help="'resume' or a gpt2 variant")
parser.add_argument('--out_dir', type=str, default='out')
parser.add_argument('--start', type=str, default='\n')
parser.add_argument('--num_samples', type=int, default=1) # Changed to 1 for easier visualization
parser.add_argument('--max_new_tokens', type=int, default=50) # Reduced for quicker testing
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top_k', type=int, default=200)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dtype', type=str, default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')
parser.add_argument('--compile', action='store_true', help='use PyTorch 2.0 to compile')
# New argument for Task 1.1
parser.add_argument('--show_probs', action='store_true', help='display a bar chart of token probabilities')
args = parser.parse_args()

# Overwrite the original config variables with parsed arguments
init_from = args.init_from
out_dir = args.out_dir
start = args.start
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k
seed = args.seed
device = args.device
dtype = args.dtype
compile = args.compile
show_probs = args.show_probs
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
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
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# Load the tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            print(f"--- Sample {k+1} ---")
            # The generation loop is now re-implemented here to allow for step-by-step analysis
            y = x
            for _ in range(max_new_tokens):
                # if the sequence context is growing too long we must crop it
                idx_cond = y if y.size(1) <= model.config.block_size else y[:, -model.config.block_size:]
                # forward the model to get the logits for the next token
                logits, _ = model(idx_cond)
                # get the logits for the last token
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # --- START: Code for Task 1.1 ---
                if show_probs:
                    # Get the top 10 probabilities and their indices
                    top_probs, top_indices = torch.topk(probs.squeeze(0), 10)
                    
                    # Convert to numpy for plotting
                    top_probs = top_probs.cpu().numpy()
                    top_indices = top_indices.cpu().numpy()

                    # Get the corresponding tokens
                    top_tokens = [decode([i]) for i in top_indices]
                    
                    # Create colors for the bar chart, default to blue
                    colors = ['blue'] * 10
                    # Check if the selected token is in the top 10
                    try:
                        # Find the index of the selected token in our top 10 list
                        selected_index = list(top_indices).index(idx_next.item())
                        # If found, set its color to red
                        colors[selected_index] = 'red'
                    except ValueError:
                        # The selected token was not in the top 10, all bars remain blue
                        pass

                    # Plotting
                    plt.figure(figsize=(12, 6))
                    plt.bar(top_tokens, top_probs, color=colors)
                    plt.xlabel('Tokens')
                    plt.ylabel('Probability')
                    
                    # Add the current text as the title
                    current_text = decode(y[0].tolist())
                    plt.title(f'Top 10 Next Token Probabilities for: "{current_text}"')
                    
                    plt.show()
                # --- END: Code for Task 1.1 ---
                
                # append sampled index to the running sequence and continue
                y = torch.cat((y, idx_next), dim=1)

            print(decode(y[0].tolist()))
            print('---------------')