# cross_gen.py
import torch
from model import GPTConfig, GPT
import tiktoken
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        new_state_dict[new_k] = v
    return new_state_dict

# Load model A
ckpt_a = torch.load("out-kant/ckpt.pt", map_location=device)
config_a = GPTConfig(**ckpt_a['model_args'])
model_a = GPT(config_a)
state_dict_a = clean_state_dict(ckpt_a['model'])
model_a.load_state_dict(state_dict_a)
model_a.to(device)
model_a.eval()

# Load model B
ckpt_b = torch.load("out-fairytails/ckpt.pt", map_location=device)
config_b = GPTConfig(**ckpt_b['model_args'])
model_b = GPT(config_b)
state_dict_b = clean_state_dict(ckpt_b['model'])
model_b.load_state_dict(state_dict_b)
model_b.to(device)
model_b.eval()

enc = tiktoken.get_encoding("gpt2")

def generate_with_switching(prompt, n_tokens=200, window=20):
    tokens = enc.encode(prompt)
    out_text = ""
    current_model = "A"
    total_generated = 0

    while total_generated < n_tokens:
        model = model_a if current_model == "A" else model_b

        x = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]
        y = model.generate(x, max_new_tokens=50, temperature=0.8, top_k=50)
        new_tokens = y[0].tolist()[len(tokens):]
        tokens += new_tokens
        total_generated += len(new_tokens)

        out_text += enc.decode(tokens)
        print(f"\n--- Model {current_model} ---\n")
        print(enc.decode(new_tokens))

        tokens = tokens[-window:]  
        current_model = "B" if current_model == "A" else "A"

    return out_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-model text generation")
    parser.add_argument("--prompt", type=str, required=True, help="Starting prompt text")
    parser.add_argument("--n_tokens", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--window", type=int, default=20, help="Token window size for switching")

    args = parser.parse_args()

    final_text = generate_with_switching(args.prompt, args.n_tokens, args.window)

    print("\n===== Final Blended Text =====\n")
    print(final_text)
