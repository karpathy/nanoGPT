# fmt: off

"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

def generate_sample(
        model, start="\n", max_new_tokens=50, temperature=0.8, top_k=200,
        device='cuda', meta_path=None):
    # Load meta if available
    if meta_path:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
    else:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={""})
        decode = lambda l: enc.decode(l)

    # Encode the beginning of the prompt
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Run generation
    generator = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    for token in generator:
        yield decode([token])


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = 'out' # ignored if init_from is not 'resume'
    start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    streaming = True # print tokens one by one
    num_samples = 10 # number of samples to draw
    max_new_tokens = 500 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    exec(open('configurator.py').read()) # overrides from command line or config file
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
    if load_meta:
        print(f"Loading meta from {meta_path}...")
    else:
        # ok let's assume gpt-2 encodings by default
        meta_path=None
        print("No meta.pkl found, assuming GPT-2 encodings...")

    # run generation
    with torch.no_grad():
        with ctx:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            for k in range(num_samples):
                try:
                    generator = generate_sample(
                        model=model,
                        start=start,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        device=device,
                        meta_path=meta_path,
                    )
                    if streaming:
                        print(start, end='')
                        for token in generator:
                            print(token, end='', flush=True)
                    else:
                        print("".join(generator), end='')
                    print("...")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                except KeyboardInterrupt:
                    break
