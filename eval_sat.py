"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
output_file = None
prompts_file = "data/SAT/SAT_test.txt"
eval_labels = None
igore_tokens = ["[PAD]"]
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.01 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
eval = True
sep = ' ' # separator between tokens during decoding, default: nothing, i.e. join with empty string
exec(open('configurator.py').read()) # overrides from command line or config file
if sep == 'SPACE':
    sep = ' '
if sep == 'EMPTY':
    sep = ''
if sep == 'NEWLINE':
    sep = '\n'
if sep == 'TAB':
    sep = '\t'
# -----------------------------------------------------------------------------

assert eval_labels is None or eval

def load_sat_tokens(fn, stoi):
    with open(fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = None
            if line.strip().endswith('UNSAT'):
                label = False
            elif line.strip().endswith('SAT'):
                label = True
            assert label is not None or eval is False
            line = line.strip().replace('-', '- ')
            if '[SEP]' in line:
                line = line[:line.index('[SEP]') + len('[SEP]')]
            elif not line.endswith('[SEP]'):
                line += ' [SEP]'
            yield [stoi[c] for c in line.split()], label

def line_sat(line, sep=' '):
    assert not ((sep + 'UNSAT') in line and (sep + ' SAT') in line) or sep == ''
    if sep + 'UNSAT' in line:
        return False
    elif sep + 'SAT' in line:
        return True
    else:
        return None

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
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: sep.join([itos[i] for i in l if itos[i] not in igore_tokens])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
# if start.startswith('FILE:'):
#     with open(start[5:], 'r', encoding='utf-8') as f:
#         start = f.read()
# start_ids = encode(start)
# x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
sample_cnt = 0
true_label = []
pred_label = []
with torch.no_grad():
    with ctx:
        for prompt, label in load_sat_tokens(prompts_file, stoi):
            if sample_cnt >= num_samples:
                break
            if eval_labels is not None:
                true_label.append(label)
                continue
            sample_cnt += 1
            prompt = (torch.tensor(prompt, dtype=torch.long, device=device)[None, ...])
            y = model.generate(prompt, max_new_tokens, temperature=temperature, top_k=top_k)
            res_str = decode(y[0].tolist())
            print(res_str)
            print('---------------')
            if eval:
                true_label.append(label)
                res = line_sat(res_str, sep)
                if res is None:
                    res = not label
                pred_label.append(res)
            if output_file is not None:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(res_str + '\n')

if eval_labels is not None:
    with open(eval_labels, 'r') as f:
        for line in f:
            pred_label.append(line_sat(line, sep))

if len(true_label) != len(pred_label):
    true_label = true_label[:len(pred_label)]

for i in range(len(true_label)):
    if pred_label[i] is None:
        pred_label[i] = not true_label[i] # If the model fails to predict, we assume it is wrong

print(true_label)
print(pred_label)
if eval:
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    f1 = f1_score(true_label, pred_label, pos_label=False)
    acc = accuracy_score(true_label, pred_label)
    prec = precision_score(true_label, pred_label, pos_label=False)
    recall = recall_score(true_label, pred_label, pos_label=False)
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
