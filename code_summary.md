# nanoGPT Core Implementation

## Overview
This directory contains the core implementation of nanoGPT, a minimal and educational implementation of GPT-style transformer language models. The codebase is designed for clarity and simplicity, making it easy to understand transformer architectures and train small to medium-sized language models from scratch.

## Purpose
Provide a clean, hackable implementation of GPT that can:
- Train GPT models from scratch or fine-tune on custom datasets
- Load and use pretrained OpenAI GPT-2 models
- Generate text samples from trained models
- Benchmark model performance
- Support both single-GPU and distributed training (DDP)

## Key Files

### `model.py` (331 lines)
**Core transformer implementation** - Complete GPT model architecture in a single file.

Components:
- `GPTConfig`: Dataclass for model configuration (block_size, vocab_size, n_layer, n_head, n_embd, dropout, bias)
- `GPT`: Main model class with transformer architecture
- `CausalSelfAttention`: Multi-head self-attention with Flash Attention support (PyTorch >= 2.0)
- `MLP`: Feed-forward network with GELU activation
- `Block`: Transformer block (attention + MLP with residual connections)
- `LayerNorm`: Custom LayerNorm with optional bias

Key methods:
- `GPT.from_pretrained()`: Load OpenAI GPT-2 weights (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- `GPT.generate()`: Autoregressive text generation with temperature and top-k sampling
- `GPT.configure_optimizers()`: Create AdamW optimizer with weight decay groups
- `GPT.estimate_mfu()`: Estimate model FLOPs utilization (for A100)

### `train.py` (337 lines)
**Training script** - Supports single-GPU and distributed data parallel (DDP) training.

Features:
- Multiple initialization modes: 'scratch', 'resume', or 'gpt2*' (pretrained)
- Distributed training via `torchrun` (DDP with NCCL/Gloo backend)
- Mixed-precision training (float16/bfloat16 with GradScaler)
- Cosine learning rate decay with linear warmup
- Gradient accumulation for simulating larger batch sizes
- Model compilation via PyTorch 2.0 `torch.compile()`
- Checkpointing and resume capability
- Optional Weights & Biases logging
- Memory-mapped dataset loading (numpy memmap)

Usage examples:
```bash
# Single GPU
python train.py --batch_size=32 --compile=False

# Multi-GPU (4 GPUs on 1 node)
torchrun --standalone --nproc_per_node=4 train.py

# Resume from checkpoint
python train.py init_from=resume

# Fine-tune GPT-2
python train.py init_from=gpt2-medium
```

### `sample.py` (90 lines)
**Text generation script** - Generate text samples from trained models.

Features:
- Load from checkpoint or pretrained GPT-2 models
- Configurable sampling parameters (temperature, top_k, num_samples, max_new_tokens)
- Automatic tokenizer detection (custom or GPT-2 tiktoken)
- Support for file-based prompts (`FILE:prompt.txt`)
- Mixed-precision inference support

Usage:
```bash
# Sample from checkpoint
python sample.py --init_from=resume --num_samples=5 --max_new_tokens=100

# Sample from GPT-2
python sample.py --init_from=gpt2-xl --start="Once upon a time"
```

### `configurator.py` (48 lines)
**Configuration override system** - Simple command-line and file-based configuration.

Design philosophy: Avoid complex configuration frameworks by using Python's `globals()` and `exec()`.

Usage pattern:
```bash
# Override with config file then command line
python train.py config/train_shakespeare.py --batch_size=32 --learning_rate=1e-4
```

Mechanism:
1. Load and execute configuration file (Python script)
2. Parse `--key=value` arguments from command line
3. Override global variables with literal_eval for type safety

### `bench.py` (118 lines)
**Benchmarking script** - Simplified training loop for performance testing.

Features:
- Measure iterations per second and model FLOPs utilization (MFU)
- Optional PyTorch profiler integration (generates TensorBoard traces)
- Configurable data loading (real data vs random data)
- Model compilation support

Usage:
```bash
# Basic benchmark
python bench.py

# With profiling
python bench.py --profile=True

# Random data (faster, no I/O)
python bench.py --real_data=False
```

## Dependencies

### Core dependencies:
- **PyTorch** >= 2.0 (for Flash Attention and `torch.compile()`)
- **NumPy** (for memory-mapped data loading)
- **tiktoken** (for GPT-2 tokenization)
- **transformers** (optional, for loading GPT-2 pretrained weights)

### Optional dependencies:
- **wandb** (Weights & Biases logging)
- **CUDA** (GPU training support)

## Relationships with Other Components

### Data Pipeline
- Training scripts expect preprocessed binary files in `data/<dataset>/`:
  - `train.bin`, `val.bin` (numpy uint16 memory-mapped arrays)
  - `meta.pkl` (vocabulary and encoding/decoding functions)
- See `data/*/prepare.py` for dataset preparation scripts

### Configuration Files
- `config/` directory contains training configurations for different model sizes and datasets
- Each config file overrides default hyperparameters in `train.py`
- Examples: `train_gpt2.py`, `train_shakespeare_char.py`, `eval_gpt2.py`

### Model Checkpoints
- Checkpoints saved to `out/` directory (configurable)
- Checkpoint format: `{'model': state_dict, 'optimizer': state_dict, 'model_args': dict, 'iter_num': int, 'best_val_loss': float, 'config': dict}`
- Compatible with `torch.load()` and `torch.save()`

## Architecture Highlights

### Flash Attention
Automatically uses Flash Attention (PyTorch >= 2.0) for ~2-3x speedup in attention computation.

### Weight Tying
Token embedding weights are tied with the output layer (`lm_head`) for improved parameter efficiency.

### Model Surgery
`GPT.crop_block_size()` allows reducing context length after loading pretrained models.

## Performance Optimizations

1. **Flash Attention**: Efficient CUDA kernels for self-attention
2. **torch.compile()**: JIT compilation for ~30% speedup
3. **Mixed Precision**: bfloat16/float16 training with automatic scaling
4. **Fused AdamW**: CUDA-fused optimizer operations
5. **TF32**: Enabled for matmul and cuDNN operations
6. **Gradient Accumulation**: Simulate larger batch sizes with limited memory
7. **Memory-mapped Data**: Zero-copy data loading with numpy memmap

## Usage Notes

- Default configuration trains a 124M parameter GPT-2 model on OpenWebText
- Requires ~40GB GPU memory for full GPT-2 (124M) training with default settings
- Supports Apple Silicon via `device='mps'` (experimental)
- DDP training requires proper network setup (Infiniband recommended, or set `NCCL_IB_DISABLE=1`)

## Educational Value

This implementation prioritizes:
- **Readability**: Complete model in a single file with clear variable names
- **Hackability**: Easy to modify and experiment with architecture changes
- **Transparency**: No hidden abstractions or complex frameworks
- **Performance**: Production-grade optimizations (Flash Attention, torch.compile, DDP)

Ideal for learning transformer architectures, training dynamics, and scaling laws.
