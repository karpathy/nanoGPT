"""
Sample generations from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import sys

from model import GPTConfig, GPT
from utils import HiddenPrints, printok, printerr

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'inital_test_6k' # ignored if init_from is not 'resume'
start = "A most notable" # or "<|endoftext|>" or etc. Can also specify a file, use the following: "FILE:prompt.txt" where 'prompt.txt' is the actual filename
num_samples = 10 # number of samples to draw
max_new_tokens = 12 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
calculate_perplexity = True #CKG added, should figure out how to calc ppl @ generation time
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
    # init from a model saved in a specific directory
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
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
# originale method of loading in a character level tokenizer from a meta.pkl
if (load_meta and "shakespeare_insults" not in meta_path):
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
# Alt method, for work with the insults & the custom tokenizer class
elif (load_meta and "shakespeare_insults" in meta_path):
    print(f"Loading in the 'shakespeare_insults' tokenizer, from {meta_path}...")
    modpath = meta_path.replace("meta.pkl", "") #dir which contains the tokenizer data + class def
    tkznrpath = meta_path.replace("meta.pkl", "insults-tokenizer.pkl")
    # this should not see the light of the good lord
    with HiddenPrints():
        sys.path.insert(1, modpath)
        from prepare import tokenizer
        # print(tokenizer.__dict__)
    # tokenization funcs, taken from the upstream class definition
    encode = tokenizer.__call__
    decode = tokenizer.decode
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    # x shape: [1, seq_len]


# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            x = x[:, :]
            print("input shape: ", x.shape)

            # optionally calculate ppl @ each decoding step during inference
            if calculate_perplexity:
                y, seq_logprobs = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, return_token_logprobs=True)
                printerr(seq_logprobs)
                output_strings = decode(y[0].tolist())
                ppl = torch.exp(-torch.sum(seq_logprobs)/len(seq_logprobs))

                # Calculate PPL @ Steps in Seq
                for idx, tok_id in enumerate(y[0]):
                    # logits[:, idx, tok_id] #alleged idx'ing of token score
                    sl_at_idx = seq_logprobs[:idx]
                    ppl_at_idx = torch.exp(-torch.sum(sl_at_idx)/len(sl_at_idx))

                    token_str = decode([tok_id.item()])[0]
                    print("".join(output_strings[:idx]), sep="", end="")
                    printok(token_str if token_str != " " else "_")
                    if idx+1 < len(y[0]):
                        next_token = y[0, idx+1]
                        print(f"  current token: {tok_id} should be followed by token: {next_token} which is '{decode([next_token.item()])[0]}' ")
                    else:
                        print(f"  final token: {tok_id} --> {y[0, idx]}")
                    # print(f"   min -- token: {worst_idx} score: {worst} str: {worst_str}")
                    # print(f"   max -- token: {best_idx} score: {best} str: {best_str}")
                    print(f"  ppl @ this point: {ppl_at_idx}")
                    print("\n")

            # Default Inference Behavior
            else: 
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, return_token_logprobs=False)
                output_strings = decode(y[0].tolist())


            print(output_strings)
            print('---------------')
            exit(42) #ckg added, early stop
