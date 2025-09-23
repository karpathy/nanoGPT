import os
import json
import torch
import tiktoken
from model_modified import GPTConfig, GPT # Imports from your modified model.py
from contextlib import nullcontext
import argparse

def eval_model(model, data_path='eval_data.json', device='cpu'):
    """
    Reads prompt-response pairs from a JSON file, calculates the probability
    of each response given the prompt, and prints the sum of probabilities.
    """
    print(f"Loading evaluation data from {data_path}...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {data_path}: {e}")
        return

    # Set up the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    total_prob = 0.0
    print("\n--- Starting Evaluation ---")
    for i, item in enumerate(eval_data):
        prompt = item['prompt']
        response = item['response']

        prompt_ids = encode(prompt)
        response_ids = encode(response)
        
        prompt_tensor = (torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...])
        response_tensor = (torch.tensor(response_ids, dtype=torch.long, device=device)[None, ...])

        # Call the generate function with our fixed_response
        with torch.no_grad():
            _, prob = model.generate(prompt_tensor, max_new_tokens=0, fixed_response=response_tensor, temperature=1.0)
            print(f"Pair {i+1}: P(Response|Prompt) = {prob:.6e}")
            total_prob += prob

    print("--- Evaluation Complete ---")
    print(f"\nSummed Probability of all pairs: {total_prob}\n")


if __name__ == '__main__':
    # --- Argument Parsing and Model Loading (adapted from sample.py) ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_from', type=str, default='gpt2', help="'resume' or a gpt2 variant")
    parser.add_argument('--out_dir', type=str, default='out', help="Output directory for resumed models")
    parser.add_argument('--device', type=str, default='cpu', help="e.g. 'cpu', 'cuda', 'cuda:0'")
    parser.add_argument('--dtype', type=str, default='float32', help="'float32', 'bfloat16', or 'float16'")
    parser.add_argument('--compile', action='store_true', help='use PyTorch 2.0 to compile')
    args = parser.parse_args()

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load the model
    if args.init_from == 'resume':
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # Fix for state dict keys
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif args.init_from.startswith('gpt2'):
        model = GPT.from_pretrained(args.init_from, dict(dropout=0.0))

    model.eval()
    model.to(args.device)
    if args.compile:
        model = torch.compile(model)

    # Run the evaluation function
    eval_model(model, device=args.device)