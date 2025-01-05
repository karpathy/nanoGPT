import os
import torch
import argparse
from model_ext import GPT, GPTConfig
from sat_dataset import SATTokenizer  # adjust this import as necessary

def main():
    parser = argparse.ArgumentParser(description="Load a trained GPT model and interactively generate text.")
    parser.add_argument('model_dir', type=str, help='Path to the folder containing ckpt.pt and model configuration.')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Number of tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for generation. 0.0 means greedy decoding.')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling. None means no top-k filtering.')
    args = parser.parse_args()

    ckpt_path = os.path.join(args.model_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)

    # Initialize the model
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print(model.summary())
    for name, param in model.named_parameters():
        print(name, param.shape, param.numel())

    # Initialize the tokenizer
    tokenizer = SATTokenizer()

    # Interactive loop
    print("Type 'quit' to exit.")
    while True:
        user_input = input("Enter your prompt: ")
        if user_input.strip().lower() == 'quit':
            break

        # Tokenize user input
        input_ids = tokenizer.encode(user_input)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        stop_tokens = tokenizer.encode("SAT UNSAT")

        # Generate output
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop=stop_tokens
        )

        # Decode and print result
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        print("Generated:", generated_text)

if __name__ == "__main__":
    main()
