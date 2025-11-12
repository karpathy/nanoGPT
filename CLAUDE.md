# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

nanoGPT is a minimalist GPT training framework designed for simplicity and hackability. It reproduces GPT-2 results while maintaining readable code (~300 lines each for training and model definition).

## Architecture

### Core Components
- **train.py**: Main training loop with distributed training support
- **model.py**: GPT model implementation with transformer architecture
- **sample.py**: Text generation/inference from trained models
- **configurator.py**: Configuration management system
- **config/**: Configuration files for different training scenarios

### Key Files
- `model.py:GPT` - Main GPT model class
- `model.py:CausalSelfAttention` - Multi-head attention implementation
- `train.py:train()` - Core training loop (around line 200+)
- `sample.py:generate()` - Text generation function

## Common Commands

### Installation
```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### Quick Start (Shakespeare Character-level)
```bash
# Prepare data
python data/shakespeare_char/prepare.py

# Train model
python train.py config/train_shakespeare_char.py

# Generate samples
python sample.py --out_dir=out-shakespeare-char
```

### GPT-2 Reproduction
```bash
# Prepare OpenWebText data
python data/openwebtext/prepare.py

# Train GPT-2 (124M) on 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

### Evaluation
```bash
# Evaluate pre-trained GPT-2 models
python train.py config/eval_gpt2.py
python train.py config/eval_gpt2_medium.py
python train.py config/eval_gpt2_large.py
python train.py config/eval_gpt2_xl.py
```

### Finetuning
```bash
# Finetune GPT-2 on Shakespeare
python train.py config/finetune_shakespeare.py
```

### Sampling from Pre-trained Models
```bash
# Sample from GPT-2 XL
python sample.py --init_from=gpt2-xl --start="Your prompt here" --num_samples=5 --max_new_tokens=100
```

### Benchmarking
```bash
# Profile training performance
python bench.py
```

## Configuration System

Configuration is handled through Python files in the `config/` directory. Key parameters:
- `batch_size`: Micro-batch size per GPU
- `gradient_accumulation_steps`: Simulates larger batch sizes
- `block_size`: Context length (sequence length)
- `n_layer`, `n_head`, `n_embd`: Model architecture parameters
- `learning_rate`, `max_iters`, `lr_decay_iters`: Training hyperparameters
- `init_from`: 'scratch', 'resume', or 'gpt2*' for initialization
- `use_alibi`: True/False - Enable ALiBi (Attention with Linear Biases) for length extrapolation

## Distributed Training

### Single GPU
```bash
python train.py --batch_size=32 --compile=False
```

### Multi-GPU (Single Node)
```bash
torchrun --standalone --nproc_per_node=4 train.py
```

### Multi-Node
```bash
# Master node
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=IP --master_port=1234 train.py

# Worker node
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=IP --master_port=1234 train.py
```

## Data Preparation

### Shakespeare Character-level
- Uses character-level tokenization
- Creates `train.bin` and `val.bin` files
- Fast setup for experimentation

### OpenWebText
- Uses GPT-2 BPE tokenization via tiktoken
- Large dataset requiring significant download time
- Used for GPT-2 reproduction

### Custom Datasets
- Follow patterns in `data/shakespeare/` or `data/openwebtext/`
- Implement `prepare.py` script for tokenization
- Output `train.bin` and `val.bin` files

## Model Configurations

### Baby GPT (Shakespeare)
- 6 layers, 6 heads, 384 embedding dim
- Context length: 256 characters
- ~3 minutes training on A100

### GPT-2 Reproduction
- 124M parameters: 12 layers, 12 heads, 768 embedding dim
- 350M, 774M, 1558M variants available
- Context length: 1024 tokens

## Device Support

### GPU (CUDA)
- Default configuration
- Use `--device=cuda` explicitly if needed

### CPU
- Add `--device=cpu --compile=False`
- Reduce model size and batch size for feasibility

### Apple Silicon (MPS)
- Add `--device=mps` for Metal GPU acceleration
- 2-3x speedup over CPU on Apple Silicon

## Troubleshooting

### PyTorch 2.0 Issues
- Add `--compile=False` to disable torch.compile
- This will slow down training but improve compatibility

### Memory Issues
- Reduce `batch_size` or `block_size`
- Use smaller model configurations
- Enable gradient accumulation instead of larger batch sizes

### Multi-GPU Issues
- Ensure proper torchrun usage
- Check NCCL configuration for multi-node setups
- Prepend `NCCL_IB_DISABLE=1` if no Infiniband

## No Testing Framework

This repository does not include a formal testing framework. Validation is performed through:
- Model evaluation on validation sets
- Loss monitoring during training
- Generated text quality assessment
- Comparison with baseline GPT-2 results

## ALiBi (Attention with Linear Biases) Support

ALiBi enables models to extrapolate to longer sequences at inference time than they were trained on.

### Key Features
- **Length Extrapolation**: Models trained on sequence length N can handle sequences of length 2N+ at inference
- **No Positional Embeddings**: ALiBi replaces standard positional embeddings with attention biases
- **Memory Efficient**: Uses linear biases instead of learned position embeddings
- **Causal Masking**: Maintains causal attention patterns

### Configuration
```python
use_alibi = True  # Enable ALiBi in model config
```

### ALiBi Training Examples
```bash
# Train character-level Shakespeare model with ALiBi
python train.py config/train_shakespeare_char_alibi.py

# Train GPT-2 with ALiBi
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_alibi.py
```

### Testing Length Extrapolation
```bash
# Test ALiBi extrapolation capabilities
python sample_alibi_extrapolation.py

# Test with trained model
python sample_alibi_extrapolation.py out-shakespeare-char-alibi
```

### Implementation Details
- ALiBi slopes are computed automatically based on number of heads
- Biases are applied directly to attention scores before softmax
- Flash Attention is disabled when using ALiBi
- Models can handle sequences longer than `block_size` at inference

## Logging and Monitoring

### Weights & Biases
- Set `wandb_log = True` in config
- Configure `wandb_project` and `wandb_run_name`
- Automatic logging of losses, learning rates, and metrics

### Local Logging
- Training progress printed to console
- Checkpoints saved to `out_dir`
- Loss curves and metrics tracked