import torch
import tiktoken
from model_modified import GPTConfig, GPT

# --- Configuration ---
device = 'cpu'
init_from = 'gpt2'
prompt = 'I live in'
# The specific sequence we want to find the probability of
fixed_response_text = ' New York City.'

# --- Model Loading ---
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
model.eval()
model.to(device)

# --- Tokenization ---
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

prompt_ids = encode(prompt)
response_ids = encode(fixed_response_text)
x = (torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...])
fixed_response_tensor = (torch.tensor(response_ids, dtype=torch.long, device=device)[None, ...])

# --- Probability Calculation for the Fixed Sequence ---
with torch.no_grad():
    # Call generate with the fixed_response argument
    full_sequence, prob = model.generate(x, max_new_tokens=0, fixed_response=fixed_response_tensor)
    
    print(f"Prompt: '{prompt}'")
    print(f"Fixed Response: '{fixed_response_text}'")
    print(f"Probability of this sequence: {prob}")