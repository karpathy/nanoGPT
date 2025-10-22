"""
Attention Pattern Visualization and Analysis

This script visualizes and analyzes attention patterns from GPT models,
particularly useful for comparing standard attention vs attention sink mechanism.

Based on the paper "Efficient Streaming Language Models with Attention Sinks"
"""

import os
import pickle
import numpy as np
import torch
from contextlib import nullcontext
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from model import GPTConfig, GPT

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False  # torch.compile can interfere with attention extraction

def load_model(checkpoint_path):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint['model_args']
    gptconf = GPTConfig(**config)
    model = GPT(gptconf)

    # Load state dict
    state_dict = checkpoint['model']
    # Fix state dict keys if necessary
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model, gptconf

def extract_attention_patterns(model, input_ids, layer_idx=0):
    """
    Extract attention patterns from a specific layer

    Args:
        model: GPT model
        input_ids: Input token IDs (B, T)
        layer_idx: Which transformer layer to extract from (0 to n_layer-1)

    Returns:
        attention_weights: (B, n_head, T, T) attention weights
    """
    model.eval()

    # We need to modify the forward pass to capture attention
    # For simplicity, we'll hook into the attention module
    attention_weights = []

    def attention_hook(module, input, output):
        # For AttentionSinkCausalSelfAttention with return_attention
        if isinstance(output, tuple):
            attention_weights.append(output[1])  # Second element is attention

    # Register hook on the target layer's attention module
    target_block = model.transformer.h[layer_idx]
    hook = target_block.attn.register_forward_hook(attention_hook)

    with torch.no_grad():
        # Forward pass
        model(input_ids)

    hook.remove()

    if attention_weights:
        return attention_weights[0]
    else:
        return None

def visualize_attention_pattern(attention_weights, save_path=None, title="Attention Pattern",
                                sink_size=None, window_size=None):
    """
    Visualize attention pattern as a heatmap

    Args:
        attention_weights: (n_head, T, T) or (T, T) attention weights
        save_path: Path to save the figure
        title: Title for the plot
        sink_size: If provided, mark the sink region
        window_size: If provided, mark the window region
    """
    if len(attention_weights.shape) == 3:
        # Average over heads
        attn = attention_weights.mean(dim=0).cpu().numpy()
    else:
        attn = attention_weights.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot attention heatmap
    im = ax.imshow(attn, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # Mark sink and window regions if provided
    if sink_size is not None and window_size is not None:
        T = attn.shape[0]
        # Draw sink region boundary (vertical line)
        ax.axvline(x=sink_size - 0.5, color='red', linestyle='--', linewidth=2,
                   label=f'Sink boundary (size={sink_size})')
        # Draw window diagonal
        for i in range(sink_size, T):
            window_start = max(sink_size, i - window_size + 1)
            # Mark the window boundary
            if window_start > sink_size:
                ax.plot([window_start - 0.5, window_start - 0.5],
                       [i - 0.5, i + 0.5], 'r-', linewidth=0.5, alpha=0.3)

        ax.legend(loc='upper right')

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    else:
        plt.show()

    plt.close()

def compute_attention_statistics(attention_weights, sink_size=4):
    """
    Compute statistics about attention distribution

    Args:
        attention_weights: (B, n_head, T, T) or (n_head, T, T) attention weights
        sink_size: Number of sink tokens

    Returns:
        Dictionary of statistics
    """
    if len(attention_weights.shape) == 4:
        # Average over batch
        attn = attention_weights.mean(dim=0)
    else:
        attn = attention_weights

    # Average over heads
    attn_avg = attn.mean(dim=0)  # (T, T)
    T = attn_avg.shape[0]

    stats = {}

    # Attention to sink tokens (first sink_size positions)
    if T > sink_size:
        # For each query position, what % of attention goes to sink tokens?
        sink_attention = attn_avg[:, :sink_size].sum(dim=1)  # (T,)
        stats['mean_sink_attention'] = sink_attention.mean().item()
        stats['sink_attention_over_time'] = sink_attention.cpu().numpy()

        # Attention to first token specifically
        first_token_attention = attn_avg[:, 0]
        stats['mean_first_token_attention'] = first_token_attention.mean().item()
        stats['first_token_attention_over_time'] = first_token_attention.cpu().numpy()

    # Attention entropy (how focused vs distributed is attention?)
    # Higher entropy = more distributed attention
    eps = 1e-10
    attn_entropy = -(attn_avg * torch.log(attn_avg + eps)).sum(dim=1)
    stats['mean_attention_entropy'] = attn_entropy.mean().item()
    stats['attention_entropy_over_time'] = attn_entropy.cpu().numpy()

    # Attention span (average distance between query and attended keys)
    positions = torch.arange(T, device=attn_avg.device).float()
    attended_positions = (attn_avg * positions.unsqueeze(0)).sum(dim=1)  # Weighted avg position
    query_positions = torch.arange(T, device=attn_avg.device).float()
    attention_span = (query_positions - attended_positions).abs()
    stats['mean_attention_span'] = attention_span.mean().item()
    stats['attention_span_over_time'] = attention_span.cpu().numpy()

    return stats

def plot_attention_statistics(stats_standard, stats_sink, save_path=None):
    """
    Compare attention statistics between standard and sink models

    Args:
        stats_standard: Statistics dict from standard attention
        stats_sink: Statistics dict from attention sink model
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Sink attention over time
    if 'sink_attention_over_time' in stats_standard:
        ax = axes[0, 0]
        positions = np.arange(len(stats_standard['sink_attention_over_time']))
        ax.plot(positions, stats_standard['sink_attention_over_time'],
               label='Standard Attention', alpha=0.7)
        ax.plot(positions, stats_sink['sink_attention_over_time'],
               label='Attention Sink', alpha=0.7)
        ax.set_xlabel('Query Position')
        ax.set_ylabel('Attention to Sink Tokens')
        ax.set_title('Attention to Initial Tokens')
        ax.legend()
        ax.grid(alpha=0.3)

    # Plot 2: First token attention
    if 'first_token_attention_over_time' in stats_standard:
        ax = axes[0, 1]
        positions = np.arange(len(stats_standard['first_token_attention_over_time']))
        ax.plot(positions, stats_standard['first_token_attention_over_time'],
               label='Standard Attention', alpha=0.7)
        ax.plot(positions, stats_sink['first_token_attention_over_time'],
               label='Attention Sink', alpha=0.7)
        ax.set_xlabel('Query Position')
        ax.set_ylabel('Attention to First Token')
        ax.set_title('Attention to First Token (BOS)')
        ax.legend()
        ax.grid(alpha=0.3)

    # Plot 3: Attention entropy
    ax = axes[1, 0]
    positions = np.arange(len(stats_standard['attention_entropy_over_time']))
    ax.plot(positions, stats_standard['attention_entropy_over_time'],
           label='Standard Attention', alpha=0.7)
    ax.plot(positions, stats_sink['attention_entropy_over_time'],
           label='Attention Sink', alpha=0.7)
    ax.set_xlabel('Query Position')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Attention Distribution Entropy')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Attention span
    ax = axes[1, 1]
    positions = np.arange(len(stats_standard['attention_span_over_time']))
    ax.plot(positions, stats_standard['attention_span_over_time'],
           label='Standard Attention', alpha=0.7)
    ax.plot(positions, stats_sink['attention_span_over_time'],
           label='Attention Sink', alpha=0.7)
    ax.set_xlabel('Query Position')
    ax.set_ylabel('Average Attention Distance')
    ax.set_title('Attention Span')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved statistics comparison to {save_path}")
    else:
        plt.show()

    plt.close()

def analyze_models(standard_checkpoint, sink_checkpoint, prompt="Hello, how are you doing today?",
                   output_dir="attention_analysis"):
    """
    Analyze and compare attention patterns between standard and sink models

    Args:
        standard_checkpoint: Path to standard model checkpoint
        sink_checkpoint: Path to attention sink model checkpoint
        prompt: Text prompt to analyze
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    print("Loading standard model...")
    model_standard, config_standard = load_model(standard_checkpoint)

    print("Loading attention sink model...")
    model_sink, config_sink = load_model(sink_checkpoint)

    # Encode prompt
    # For simplicity, using character-level encoding (Shakespeare dataset style)
    # In practice, you'd use the proper tokenizer
    # Here we'll just create a simple token sequence
    torch.manual_seed(42)
    sequence_length = 256
    input_ids = torch.randint(0, min(config_standard.vocab_size, config_sink.vocab_size),
                              (1, sequence_length), device=device)

    print(f"Analyzing sequence of length {sequence_length}...")

    # Extract attention patterns from middle layer
    layer_idx = config_standard.n_layer // 2

    print("Extracting attention patterns from standard model...")
    attn_standard = extract_attention_patterns(model_standard, input_ids, layer_idx)

    print("Extracting attention patterns from attention sink model...")
    attn_sink = extract_attention_patterns(model_sink, input_ids, layer_idx)

    # Visualize attention patterns
    if attn_standard is not None:
        print("Visualizing standard attention...")
        visualize_attention_pattern(
            attn_standard[0] if attn_standard.dim() == 4 else attn_standard,
            save_path=os.path.join(output_dir, "attention_standard.png"),
            title=f"Standard Attention (Layer {layer_idx})"
        )

        # Compute and print statistics
        stats_standard = compute_attention_statistics(attn_standard,
                                                     sink_size=config_sink.sink_size)
        print("\nStandard Attention Statistics:")
        print(f"  Mean attention to sink region: {stats_standard.get('mean_sink_attention', 'N/A'):.3f}")
        print(f"  Mean attention to first token: {stats_standard.get('mean_first_token_attention', 'N/A'):.3f}")
        print(f"  Mean attention entropy: {stats_standard['mean_attention_entropy']:.3f}")
    else:
        print("Could not extract attention from standard model (using Flash Attention)")
        stats_standard = None

    if attn_sink is not None:
        print("\nVisualizing attention sink...")
        visualize_attention_pattern(
            attn_sink[0] if attn_sink.dim() == 4 else attn_sink,
            save_path=os.path.join(output_dir, "attention_sink.png"),
            title=f"Attention Sink (Layer {layer_idx})",
            sink_size=config_sink.sink_size,
            window_size=config_sink.window_size
        )

        # Compute and print statistics
        stats_sink = compute_attention_statistics(attn_sink,
                                                  sink_size=config_sink.sink_size)
        print("\nAttention Sink Statistics:")
        print(f"  Mean attention to sink region: {stats_sink.get('mean_sink_attention', 'N/A'):.3f}")
        print(f"  Mean attention to first token: {stats_sink.get('mean_first_token_attention', 'N/A'):.3f}")
        print(f"  Mean attention entropy: {stats_sink['mean_attention_entropy']:.3f}")
    else:
        print("Could not extract attention from sink model (using Flash Attention)")
        stats_sink = None

    # Compare statistics if both available
    if stats_standard is not None and stats_sink is not None:
        print("\nPlotting comparison...")
        plot_attention_statistics(
            stats_standard, stats_sink,
            save_path=os.path.join(output_dir, "attention_comparison.png")
        )

    print(f"\nAnalysis complete! Results saved to {output_dir}/")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize and analyze attention patterns')
    parser.add_argument('--standard_checkpoint', type=str, required=True,
                       help='Path to standard model checkpoint')
    parser.add_argument('--sink_checkpoint', type=str, required=True,
                       help='Path to attention sink model checkpoint')
    parser.add_argument('--output_dir', type=str, default='attention_analysis',
                       help='Directory to save visualizations')

    args = parser.parse_args()

    analyze_models(args.standard_checkpoint, args.sink_checkpoint,
                   output_dir=args.output_dir)
