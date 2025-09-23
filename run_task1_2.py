import torch
import tiktoken
from model_modified import GPT # This will import the GPT class from your modified model.py

# --- Configuration (matches the assignment prompt) ---
device = 'cpu' # We know you're using CPU from the last error
init_from = 'gpt2'
start = 'I live in'
max_new_tokens = 5
temperature = 0.0001 # Using a very low temperature for deterministic results

# --- Model Loading ---
print("Loading model...")
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
model.eval()
model.to(device)
print("Model loaded.")

# --- Tokenization ---
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# --- Run Generation and Get Probability ---
print("Generating sequence and calculating probability...")
with torch.no_grad():
    # Your new generate function returns TWO values. We capture both here.
    generated_sequence_ids, probability = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature)
    
    # Decode the result to readable text
    full_text = decode(generated_sequence_ids[0].tolist())
    
    # Print the results
    print("\n--- RESULTS FOR TASK 1.2 ---")
    print(f"Prompt: '{start}'")
    print(f"Full Generated Sequence: '{full_text}'")
    print(f"Probability of the *generated part*: {probability:.6e}") # Format as scientific notation