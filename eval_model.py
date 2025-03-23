import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# PARAMS
temperature = 0.1
top_k=200
seed = 1337
device = 'cuda' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
out_dir = 'out-transcendence-gpt'
exec(open('configurator.py').read()) 

# Torch configs
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
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
    model = torch.compile(model)

data_dir = os.path.join('data', dataset)

test_data = torch.load(os.path.join(data_dir, 'test.pt'))[0]
correct_predictions = 0

# run generation
with torch.no_grad():
    with ctx:
        num_samples = len(test_data)

        for i in range(num_samples):
            input_sequence = test_data[i, :-2].unsqueeze(0).to(device)

            actual_next_token = test_data[i, -2]
            
            predicted_next_token = model.generate(input_sequence, 1, temperature=temperature, top_k=top_k)[0][-1]
            if predicted_next_token == actual_next_token:
                correct_predictions += 1
    

accuracy = (correct_predictions / num_samples) * 100

print(f"Accuracy over {num_samples} samples: {accuracy}")
