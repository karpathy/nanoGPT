#!/usr/bin/env python3
"""
Demonstration of ALiBi's length extrapolation capabilities.
This script shows how a model trained on sequences of length N can generate
sequences of length 2N or more at inference time.
"""

import os
import pickle
import torch
from model import GPT, GPTConfig

def load_model_and_generate(out_dir, max_length_multiplier=2.0, num_samples=3):
    """
    Load a trained ALiBi model and generate samples at different lengths.
    
    Args:
        out_dir: Directory containing the trained model
        max_length_multiplier: How much longer than training length to generate
        num_samples: Number of samples to generate
    """
    
    # Load the trained model
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Create model from checkpoint
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Load tokenizer/encoding info
    meta_path = os.path.join('data', gptconf.dataset if hasattr(gptconf, 'dataset') else 'shakespeare_char', 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # fallback to GPT-2 tokenizer
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Test different sequence lengths
    training_length = gptconf.block_size
    test_lengths = [training_length, int(training_length * 1.5), int(training_length * max_length_multiplier)]
    
    # Starting prompt
    start_prompt = "ROMEO:\nBut soft! What light through yonder window breaks?\nIt is the east, and Juliet is the sun."
    if hasattr(gptconf, 'dataset') and gptconf.dataset == 'shakespeare_char':
        start_ids = encode(start_prompt[:50])  # Truncate for character-level
    else:
        start_ids = encode(start_prompt)
    
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    print(f"Model trained on sequences of length: {training_length}")
    print(f"Using ALiBi: {gptconf.use_alibi}")
    print(f"Starting prompt: {decode(start_ids)}")
    print("=" * 80)
    
    for test_length in test_lengths:
        print(f"\nGenerating {num_samples} samples at length {test_length} (ratio: {test_length/training_length:.1f}x):")
        print("-" * 60)
        
        # Calculate how many new tokens to generate
        max_new_tokens = max(1, test_length - len(start_ids))
        
        for i in range(num_samples):
            with torch.no_grad():
                # Modify the generation method to allow longer sequences
                original_block_size = model.config.block_size
                if gptconf.use_alibi:
                    # Temporarily increase the effective block size for ALiBi models
                    model.config.block_size = test_length
                
                try:
                    y = model.generate(x, max_new_tokens, temperature=0.8, top_k=200)
                    generated_text = decode(y[0].tolist())
                    
                    print(f"Sample {i+1}:")
                    print(generated_text)
                    print()
                    
                finally:
                    # Restore original block size
                    model.config.block_size = original_block_size
    
    return model

def compare_standard_vs_alibi():
    """
    Compare standard positional embeddings vs ALiBi on length extrapolation.
    This creates two small models and shows the difference.
    """
    print("Comparing Standard vs ALiBi models on length extrapolation...")
    print("=" * 80)
    
    # Create small models for comparison
    config_std = GPTConfig(
        block_size=64,
        vocab_size=50,
        n_layer=4,
        n_head=4,
        n_embd=256,
        use_alibi=False
    )
    
    config_alibi = GPTConfig(
        block_size=64,
        vocab_size=50,
        n_layer=4,
        n_head=4,
        n_embd=256,
        use_alibi=True
    )
    
    model_std = GPT(config_std)
    model_alibi = GPT(config_alibi)
    
    # Create test sequence longer than training length
    test_seq_len = 96  # 1.5x the training length
    x = torch.randint(0, 50, (1, test_seq_len))
    
    print(f"Training length: {config_std.block_size}")
    print(f"Test sequence length: {test_seq_len}")
    print(f"Extrapolation ratio: {test_seq_len / config_std.block_size:.1f}x")
    
    # Test standard model (should fail or perform poorly)
    print("\nTesting Standard Positional Embeddings Model:")
    try:
        with torch.no_grad():
            logits_std, _ = model_std(x[:, :config_std.block_size], targets=x[:, :config_std.block_size])
            print(f"✓ Standard model works at training length: {logits_std.shape}")
            
            # This should fail or perform poorly
            try:
                logits_std_long, _ = model_std(x, targets=x)
                print(f"✗ Standard model at {test_seq_len} length: {logits_std_long.shape} (unexpected success)")
            except Exception as e:
                print(f"✓ Standard model fails at {test_seq_len} length: {str(e)}")
    except Exception as e:
        print(f"✗ Standard model failed: {str(e)}")
    
    # Test ALiBi model (should work)
    print("\nTesting ALiBi Model:")
    try:
        with torch.no_grad():
            logits_alibi, _ = model_alibi(x[:, :config_alibi.block_size], targets=x[:, :config_alibi.block_size])
            print(f"✓ ALiBi model works at training length: {logits_alibi.shape}")
            
            logits_alibi_long, _ = model_alibi(x, targets=x)
            print(f"✓ ALiBi model works at {test_seq_len} length: {logits_alibi_long.shape}")
    except Exception as e:
        print(f"✗ ALiBi model failed: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        out_dir = sys.argv[1]
        if os.path.exists(out_dir):
            print(f"Loading model from {out_dir}...")
            load_model_and_generate(out_dir)
        else:
            print(f"Directory {out_dir} does not exist.")
            print("Usage: python sample_alibi_extrapolation.py <model_output_dir>")
    else:
        print("No model directory provided. Running comparison demo...")
        compare_standard_vs_alibi()
        print("\n" + "=" * 80)
        print("To test with a trained model, run:")
        print("python sample_alibi_extrapolation.py <model_output_dir>")
        print("\nExample:")
        print("python sample_alibi_extrapolation.py out-shakespeare-char-alibi")