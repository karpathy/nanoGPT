"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from rich import print
from model import GPTConfig, GPT
import argparse

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

  return parser.parse_args()

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
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    if 'tokenizer' in meta and meta['tokenizer'] == 'tiktoken':
        enc = tiktoken.get_encoding(meta['tiktoken_encoding'])
        print(f"using tiktoken encoding {meta['tiktoken_encoding']}")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    elif 'tokenizer' in meta and meta['tokenizer'] == 'sentencepiece':

        separator_token="▁"

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
x = (torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(args.num_samples):
            y = model.generate(x, args.max_new_tokens,
                               temperature=args.temperature, top_k=args.top_k)
            if separator_token == None:
                print("[bold green]" + decode(y[0].tolist()))
                print('---------------')
            else:
                print(separator_token)
                print("[bold green]" + decode(y[0].tolist()).replace(separator_token, " "))
                print('---------------')

