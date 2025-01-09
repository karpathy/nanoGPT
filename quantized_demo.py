from quantized_model import QuantGPT
from model import GPT, GPTConfig

import tiktoken
import torch

max_new_tokens = 1000

"""
Demo of the code is available on Google Colab : https://colab.research.google.com/drive/1XNgge1sxYtfZTGfzRhqDDEiBV6BaBbe7?usp=sharing
"""

if __name__ == "__main__":
    
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    prompt = "Ones upon a time"
    start_ids = torch.tensor(encode(prompt)).unsqueeze(0)
    
    max_length = 50
    temperature = 0.9
    top_k = 40
    
    block_size = 1024
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0 
    bias = False
    
    gpt2_config = GPTConfig(**dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout))
    
    gpt_2 = GPT.from_pretrained('gpt2')
    q_gpt_2 = QuantGPT.from_pretrained('gpt2')

    print("Original GPT2:")
    y1 = gpt_2.generate(start_ids, max_new_tokens=max_length, temperature=temperature, top_k=top_k)
    print(decode(y1[0].tolist()))

    print("\nQuantized GPT2:")
    y2 = q_gpt_2.generate(start_ids, max_new_tokens=max_length, temperature=temperature, top_k=top_k)
    print(decode(y2[0].tolist()))
    
    