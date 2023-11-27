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
from visualization import display_colored_text, catpuccin_hue
from utils import HiddenPrints, printok, printerr, printlog, printwar

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'inital_test_6k' # ignored if init_from is not 'resume'
start = "A most notable" # or "<|endoftext|>" or etc. Can also specify a file, use the following: "FILE:prompt.txt" where 'prompt.txt' is the actual filename
start = "012"
# start = "92" #get jenky with this countformer
num_samples = 10 # number of samples to draw
max_new_tokens = 10 # number of tokens generated in each sample
temperature = 0.1 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
# temperature = 120
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

            # Optional Inference-ing: calculate ppl @ each decoding step
            if calculate_perplexity:
                y, output_logprobs = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, return_output_logprobs=True)
                    # output_logprobs shape: (batch, sequence_length, vocab_size)
                printlog("WE @ THEM PROBS FOE")
                printerr(output_logprobs.shape)
                for idx, lp in enumerate(output_logprobs[0]):
                    print(f"@ {y[0, idx]} -- {lp} -- next should be {lp.argmax()}")

                # attempt to calculate perplexity
                seq_logprobs = torch.tensor([0.0], device=device) #include a "logprob" for the zero-th index --> 1st token is something like oracle logprob, since MAX loglikelihood
                seq_ppls = [1.0] #placeholder for 1st pos ppl, assume perfect as above (lowest ppl 1.0)
                for idx, tok in enumerate(y[0]):
                    alleged_max = output_logprobs[0, idx, :].argmax()
                    if idx+1 < len(y[0]):
                        actual_next_token = y[0, idx+1]
                        # print(idx, " --> has alleged_max: ", alleged_max.item(), "and actual selected: ", actual_next_token.item())
                        seq_logprobs = torch.cat((seq_logprobs, output_logprobs[0, idx, actual_next_token].unsqueeze(0)), dim=0)
                        seq_ppls.append(
                            torch.exp2(-torch.sum(seq_logprobs)/len(seq_logprobs)).item()
                        )
                    # once the index is greater than y's len, there are no more "actuals" to compute ppl with...

                ppl = torch.exp2(-torch.sum(seq_logprobs)/len(seq_logprobs)) #for aggregate sequence
                printlog(ppl)
                printlog(seq_ppls)

                output_strings = decode(y[0].tolist())
                display_colored_text(output_strings, seq_ppls, cmap=catpuccin_hue, outfilename="sequence-ppl.png")

            # Default Inference Behavior
            else: 
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, return_token_logprobs=False)
                output_strings = decode(y[0].tolist())


            print(output_strings)
            print('---------------')
            # TODO: remove this early stopping logic
            exit(42) #ckg added, early stop
