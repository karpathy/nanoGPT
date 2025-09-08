# cross_gen.py
import torch
from model import GPTConfig, GPT
from sample import sample
import tiktoken

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_a = torch.load("out-kant/ckpt.pt", map_location=device)
config_a = GPTConfig(**ckpt_a['model_args'])
model_a = GPT(config_a)
model_a.load_state_dict(ckpt_a['model'])
model_a.to(device)
model_a.eval()

ckpt_b = torch.load("out-fairytails/ckpt.pt", map_location=device)
config_b = GPTConfig(**ckpt_b['model_args'])
model_b = GPT(config_b)
model_b.load_state_dict(ckpt_b['model'])
model_b.to(device)
model_b.eval()

enc = tiktoken.get_encoding("gpt2")

def generate_with_switching(prompt, n_tokens=200, window=20):
    tokens = enc.encode(prompt)
    out_text = prompt
    current_model = "A"
    total_generated = 0

    while total_generated < n_tokens:

        model = model_a if current_model == "A" else model_b

        x = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]
        y = model.generate(x, max_new_tokens=50, temperature=0.8, top_k=50)
        new_tokens = y[0].tolist()[len(tokens):]  
        tokens += new_tokens
        total_generated += len(new_tokens)

        out_text = enc.decode(tokens)
        print(f"\n--- Model {current_model} ---\n")
        print(enc.decode(new_tokens))

    
        tokens = tokens[-window:]  
        current_model = "B" if current_model == "A" else "A"

    return out_text

# Example run
final_text = generate_with_switching("Once upon a midnight dreary,", n_tokens=200, window=20)

print("\n===== Final Blended Text =====\n")
print(final_text)