"""
Evaluation script for Attention Sink experiment

This script:
1. Loads both baseline and attention sink models
2. Evaluates perplexity on validation set
3. Tests generation quality
4. Compares training efficiency metrics
5. Analyzes attention patterns
"""

import os
import pickle
import numpy as np
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT
import tiktoken

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

def load_model_and_config(checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    config = checkpoint['model_args']
    gptconf = GPTConfig(**config)
    model = GPT(gptconf)

    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Get training stats
    iter_num = checkpoint.get('iter_num', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    return model, gptconf, iter_num, best_val_loss

@torch.no_grad()
def evaluate_perplexity(model, data_path, block_size, batch_size=8, max_batches=50):
    """
    Evaluate perplexity on a dataset

    Args:
        model: GPT model
        data_path: Path to binary data file
        block_size: Context length
        batch_size: Batch size for evaluation
        max_batches: Maximum number of batches to evaluate

    Returns:
        perplexity: Perplexity score
        loss: Average loss
    """
    # Load data
    data = np.memmap(data_path, dtype=np.uint16, mode='r')

    model.eval()
    losses = []

    # Evaluate on random batches
    for _ in range(max_batches):
        # Get random batch
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

        x, y = x.to(device), y.to(device)

        # Forward pass
        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)

        losses.append(loss.item())

    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)

    return perplexity, avg_loss

@torch.no_grad()
def generate_samples(model, meta_path, num_samples=5, max_new_tokens=200,
                     temperature=0.8, top_k=200):
    """
    Generate text samples from the model

    Args:
        model: GPT model
        meta_path: Path to metadata pickle
        num_samples: Number of samples to generate
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling

    Returns:
        List of generated text samples
    """
    # Load meta information
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    stoi = meta['stoi']
    itos = meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    model.eval()
    samples = []

    # Different starting prompts
    prompts = [
        "ROMEO:",
        "First Citizen:",
        "KING HENRY IV:",
        "The ",
        "What "
    ]

    for i, prompt in enumerate(prompts[:num_samples]):
        # Encode prompt
        start_ids = encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

        # Generate
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

        # Decode
        generated_text = decode(y[0].tolist())
        samples.append(generated_text)

    return samples

def compare_models(baseline_checkpoint, sink_checkpoint, data_dir, output_file='comparison_results.txt'):
    """
    Compare baseline and attention sink models

    Args:
        baseline_checkpoint: Path to baseline model checkpoint
        sink_checkpoint: Path to attention sink model checkpoint
        data_dir: Directory containing validation data
        output_file: File to write comparison results
    """
    print("="*80)
    print("ATTENTION SINK EXPERIMENT - MODEL COMPARISON")
    print("="*80)

    # Load models
    print("\n1. Loading models...")
    baseline_model, baseline_config, baseline_iter, baseline_val_loss = load_model_and_config(baseline_checkpoint)
    sink_model, sink_config, sink_iter, sink_val_loss = load_model_and_config(sink_checkpoint)

    print(f"\nBaseline model:")
    print(f"  - Iterations trained: {baseline_iter}")
    print(f"  - Best val loss: {baseline_val_loss:.4f}")
    print(f"  - Attention type: Standard Causal Attention")

    print(f"\nAttention Sink model:")
    print(f"  - Iterations trained: {sink_iter}")
    print(f"  - Best val loss: {sink_val_loss:.4f}")
    print(f"  - Attention type: Attention Sink")
    print(f"  - Sink size: {sink_config.sink_size}")
    print(f"  - Window size: {sink_config.window_size}")

    # Evaluate perplexity
    print("\n2. Evaluating perplexity on validation set...")
    val_data_path = os.path.join(data_dir, 'val.bin')

    baseline_ppl, baseline_loss = evaluate_perplexity(
        baseline_model, val_data_path, baseline_config.block_size
    )
    print(f"Baseline - Loss: {baseline_loss:.4f}, Perplexity: {baseline_ppl:.2f}")

    sink_ppl, sink_loss = evaluate_perplexity(
        sink_model, val_data_path, sink_config.block_size
    )
    print(f"Attention Sink - Loss: {sink_loss:.4f}, Perplexity: {sink_ppl:.2f}")

    ppl_diff = ((sink_ppl - baseline_ppl) / baseline_ppl) * 100
    print(f"Perplexity difference: {ppl_diff:+.2f}%")

    # Generate samples
    print("\n3. Generating text samples...")
    meta_path = os.path.join(data_dir, 'meta.pkl')

    print("\nBaseline samples:")
    baseline_samples = generate_samples(baseline_model, meta_path)
    for i, sample in enumerate(baseline_samples, 1):
        print(f"\n  Sample {i}:")
        print(f"  {sample[:200]}...")

    print("\n\nAttention Sink samples:")
    sink_samples = generate_samples(sink_model, meta_path)
    for i, sample in enumerate(sink_samples, 1):
        print(f"\n  Sample {i}:")
        print(f"  {sample[:200]}...")

    # Model size comparison
    print("\n4. Model parameters...")
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    sink_params = sum(p.numel() for p in sink_model.parameters())
    print(f"Baseline parameters: {baseline_params:,}")
    print(f"Attention Sink parameters: {sink_params:,}")
    print(f"Parameter difference: {sink_params - baseline_params:,} ({((sink_params - baseline_params) / baseline_params * 100):+.2f}%)")

    # Write results to file
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ATTENTION SINK EXPERIMENT RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write("MODELS\n")
        f.write("-"*40 + "\n")
        f.write(f"Baseline: {baseline_checkpoint}\n")
        f.write(f"  - Training iterations: {baseline_iter}\n")
        f.write(f"  - Best validation loss: {baseline_val_loss:.4f}\n")
        f.write(f"  - Parameters: {baseline_params:,}\n\n")

        f.write(f"Attention Sink: {sink_checkpoint}\n")
        f.write(f"  - Training iterations: {sink_iter}\n")
        f.write(f"  - Best validation loss: {sink_val_loss:.4f}\n")
        f.write(f"  - Sink size: {sink_config.sink_size}\n")
        f.write(f"  - Window size: {sink_config.window_size}\n")
        f.write(f"  - Parameters: {sink_params:,}\n\n")

        f.write("EVALUATION METRICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Baseline Perplexity: {baseline_ppl:.2f}\n")
        f.write(f"Attention Sink Perplexity: {sink_ppl:.2f}\n")
        f.write(f"Perplexity difference: {ppl_diff:+.2f}%\n\n")

        f.write("GENERATED SAMPLES\n")
        f.write("-"*40 + "\n")
        f.write("Baseline:\n")
        for i, sample in enumerate(baseline_samples, 1):
            f.write(f"\nSample {i}:\n{sample}\n")

        f.write("\n\nAttention Sink:\n")
        for i, sample in enumerate(sink_samples, 1):
            f.write(f"\nSample {i}:\n{sample}\n")

    print(f"\n\nResults written to {output_file}")
    print("\n" + "="*80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate and compare attention sink experiment')
    parser.add_argument('--baseline_checkpoint', type=str,
                       default='out-baseline-sink/ckpt.pt',
                       help='Path to baseline model checkpoint')
    parser.add_argument('--sink_checkpoint', type=str,
                       default='out-attention-sink/ckpt.pt',
                       help='Path to attention sink model checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default='data/shakespeare_char',
                       help='Directory containing validation data')
    parser.add_argument('--output_file', type=str,
                       default='attention_sink_results.txt',
                       help='Output file for results')

    args = parser.parse_args()

    compare_models(args.baseline_checkpoint, args.sink_checkpoint,
                   args.data_dir, args.output_file)
