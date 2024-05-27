from torch.nn import functional as F
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from rich import print
from model import GPTConfig, GPT
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def parseargs():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--device", type=str, help="device to run inference, e.g. 'cpu' or 'cuda' or 'cuda:0', 'cuda:1', etc...")
    parser.add_argument("--out_dir", type=str, help="directory to load checkpoint from")
    parser.add_argument("--init_from", type=str, default="resume", help="Either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')")
    parser.add_argument("--start", type=str, default="\n", help="\\n or '<|endoftext|>' or etc. Can also specify a file, use as: 'FILE:prompt.txt'")
    parser.add_argument("--num_samples", type=int, default=3, help="number of inference streams to draw")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="number of tokens generated in each sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions")
    parser.add_argument("--top_k", type=int, default=200, help="retain only the top_k most likely tokens, clamp others to have 0 probability")
    parser.add_argument("--seed", type=int, default=1337, help="seed for psuedorandom number generator")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="torch data type for inference, e.g. 'int8'")
    parser.add_argument('--compile', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--sample_file', type=str, default=None, help="output file for inference")
    parser.add_argument('--interactive', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--stop_string', type=str, default='~W', help="fixed string to stop generation and allow user input")
    parser.add_argument('--show_heatmaps', default=False, action=argparse.BooleanOptionalAction, help="show heatmaps of top-k choices for each token")
    parser.add_argument('--last_k_tokens', type=int, default=10, help="number of last tokens to display in heatmaps")
    parser.add_argument('--chart_type', type=str, default='heatmap', choices=['heatmap', 'barchart'], help="type of chart to display: 'heatmap' or 'barchart'")

    return parser.parse_args()

def save_chart(probs, idx, decode, step, out_dir, last_k_tokens, chart_type, selected_token):
    top_k_probs, top_k_indices = torch.topk(probs, k=probs.size(-1))
    top_k_tokens = [decode([top_k_indices[0, i].item()]) for i in range(top_k_indices.size(1))]

    plt.figure(figsize=(10, 6))
    
    if chart_type == 'heatmap':
        sns.heatmap(top_k_probs.cpu().numpy().reshape(1, -1), annot=np.array(top_k_tokens).reshape(1, -1), fmt='', cmap='viridis')
        plt.title(f"Step {step}: Top-k Token Probabilities")
    elif chart_type == 'barchart':
        colors = sns.color_palette('viridis', len(top_k_tokens))
        bars = plt.bar(top_k_tokens, top_k_probs.cpu().numpy().flatten(), color=colors)
        plt.title(f"Step {step}: Top-k Token Probabilities")
        plt.xticks(rotation=90)
        for bar, token in zip(bars, top_k_tokens):
            if token == selected_token:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
    
    last_tokens = decode(idx[0, -last_k_tokens:].tolist())
    plt.xlabel(f"Last {last_k_tokens} Tokens: {last_tokens}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{timestamp}_step{step}.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def interactive_generation(model, start_ids, device, max_new_tokens, temperature, top_k, stop_string, decode, encode):
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    while True:
        x, generated_text = model.generate_with_stop(x, max_new_tokens, stop_string, decode, temperature, top_k)
        print("[bold green]" + generated_text)

        user_input = input("User input (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Append the user input directly after the stop string
        x = torch.cat((x, torch.tensor(encode(user_input), dtype=torch.long, device=device)[None, ...]), dim=1)

args = parseargs()
# -----------------------------------------------------------------------------

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in args.device else 'cpu' # for later use in torch.autocast
ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if args.init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    checkpoint['model_args']['dropout'] = 0.0 # typically we don't want dropout during inference
    gptconf = GPTConfig(**checkpoint['model_args'])
    print(checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif args.init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(args.init_from, dict(dropout=0.0))

model.eval()
model.to(args.device)
if args.compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
separator_token = None
if args.init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    if 'tokenizer' in meta and meta['tokenizer'] == 'tiktoken':
        enc = tiktoken.get_encoding(meta['tiktoken_encoding'])
        print(f"using tiktoken encoding {meta['tiktoken_encoding']}")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    elif 'tokenizer' in meta and meta['tokenizer'] == 'sentencepiece':
        separator_token = "▁"
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])

# encode the beginning of the prompt
if args.start.startswith('FILE:'):
    with open(args.start[5:], 'r', encoding='utf-8') as f:
        args.start = f.read()
start_ids = encode(args.start)

if args.interactive:
    # Run interactive generation
    interactive_generation(model, start_ids, args.device, args.max_new_tokens, args.temperature, args.top_k, args.stop_string, decode, encode)
else:
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(args.num_samples):
                for step in range(args.max_new_tokens):
                    idx_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :] / args.temperature
                    if args.top_k is not None:
                        v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    x = torch.cat((x, idx_next), dim=1)

                    if args.show_heatmaps:
                        selected_token = decode([idx_next[0].item()])
                        save_chart(probs, x, decode, step, args.out_dir, args.last_k_tokens, args.chart_type, selected_token)

                output_line = decode(x[0].tolist()).replace(separator_token, " ") if separator_token else decode(x[0].tolist())
                print("[bold green]" + output_line)
                print('---------------')
                if args.sample_file:
                    with open(args.sample_file, "a") as file:
                        file.write(output_line)

