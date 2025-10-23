# Configuration Files

## Overview
This directory contains Python configuration files that override default hyperparameters in `train.py` and `sample.py`. Each configuration is tailored for specific training scenarios, model sizes, or datasets.

## Purpose
Provide ready-to-use training configurations for:
- Training GPT-2 models at various scales (124M to 1558M parameters)
- Fine-tuning pretrained models on custom datasets
- Evaluating pretrained GPT-2 models
- Quick experimentation on small datasets (Shakespeare character-level)

## Configuration System
Configurations use the `configurator.py` mechanism where Python files are executed to override global variables in training scripts. This allows natural Python syntax while maintaining simplicity.

## Key Configuration Files

### Training Configurations

#### `train_gpt2.py`
**Full-scale GPT-2 (124M) training on OpenWebText**

Target: Reproduce GPT-2 124M training (loss ~2.85)
- Hardware: 8x A100 40GB GPUs
- Total batch size: ~0.5M tokens (12 batch × 1024 block × 40 grad_accum × 8 GPUs = 491,520)
- Training tokens: 300B (600K iterations)
- Training time: ~5 days on 8x A100
- Weight decay: 1e-1
- Wandb logging: Enabled (project: 'owt', run: 'gpt2-124M')

Usage:
```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

#### `train_shakespeare_char.py`
**Miniature character-level Shakespeare model**

Purpose: Debugging, learning, and experimentation on laptops/MacBooks
- Output directory: `out-shakespeare-char`
- Dataset: `shakespeare_char` (character-level tokenization)
- Model size: "Baby GPT" - 6 layers, 6 heads, 384 embedding dim
- Context: 256 characters
- Batch size: 64
- Training: 5000 iterations (~15 minutes on consumer GPU)
- Learning rate: 1e-3 (higher due to small model)
- Dropout: 0.2 (to prevent overfitting on small dataset)
- Checkpointing: Only saves when validation improves

Notes:
- Designed for quick iteration and overfitting detection
- Can run on CPU (uncomment `device='cpu'` and `compile=False`)
- Good starting point for understanding transformer training

#### `finetune_shakespeare.py`
**Fine-tune GPT-2 XL (1558M) on Shakespeare**

Purpose: Demonstrate transfer learning from pretrained GPT-2
- Base model: `gpt2-xl` (1558M parameters, largest GPT-2)
- Dataset: `shakespeare` (word-level tokenization)
- Training strategy: Constant learning rate (no decay)
- Learning rate: 3e-5 (small for fine-tuning)
- Total tokens per iteration: 32,768 (1 batch × 32 grad_accum × 1024 tokens)
- Epochs: ~2.2 epochs over Shakespeare (301,966 tokens, 20 iterations)
- Evaluation: Every 5 iterations (40 eval batches)
- Checkpointing: Only saves when validation improves

Usage:
```bash
python train.py config/finetune_shakespeare.py
```

Expected outcome: GPT-2 XL learns Shakespeare's style in ~20 iterations

### Evaluation Configurations

#### `eval_gpt2.py`
**Evaluate base GPT-2 (124M) on validation set**

Model specs:
- Parameters: 124M
- Architecture: 12 layers, 12 heads, 768 embedding dim
- Batch size: 8
- Evaluation iterations: 500 (more iterations for accurate loss estimate)
- Mode: `eval_only=True` (exits after evaluation)
- Logging: Wandb disabled

Usage:
```bash
python train.py config/eval_gpt2.py
```

#### `eval_gpt2_medium.py`
**Evaluate GPT-2 Medium (350M)**

Model specs:
- Parameters: 350M
- Architecture: 24 layers, 16 heads, 1024 embedding dim

#### `eval_gpt2_large.py`
**Evaluate GPT-2 Large (774M)**

Model specs:
- Parameters: 774M
- Architecture: 36 layers, 20 heads, 1280 embedding dim

#### `eval_gpt2_xl.py`
**Evaluate GPT-2 XL (1558M)**

Model specs:
- Parameters: 1558M
- Architecture: 48 layers, 25 heads, 1600 embedding dim

## Configuration Parameters

### Common Overridable Parameters

**I/O and Logging:**
- `out_dir`: Output directory for checkpoints
- `wandb_log`: Enable Weights & Biases logging
- `wandb_project`, `wandb_run_name`: W&B project configuration
- `eval_interval`: Steps between evaluations
- `eval_iters`: Number of batches for evaluation
- `log_interval`: Steps between training logs
- `always_save_checkpoint`: Save checkpoint at every eval (vs only when val improves)

**Data and Batching:**
- `dataset`: Dataset name (corresponds to `data/<dataset>/`)
- `batch_size`: Per-GPU batch size
- `block_size`: Context length (sequence length)
- `gradient_accumulation_steps`: Accumulate gradients to simulate larger batches

**Model Architecture:**
- `n_layer`: Number of transformer layers
- `n_head`: Number of attention heads
- `n_embd`: Embedding dimensionality
- `dropout`: Dropout probability
- `bias`: Use bias in Linear and LayerNorm layers

**Training:**
- `learning_rate`: Peak learning rate
- `max_iters`: Total training iterations
- `weight_decay`: AdamW weight decay
- `beta1`, `beta2`: AdamW beta parameters
- `grad_clip`: Gradient clipping value

**Learning Rate Schedule:**
- `decay_lr`: Enable cosine LR decay
- `warmup_iters`: Linear warmup steps
- `lr_decay_iters`: Total decay steps
- `min_lr`: Minimum learning rate after decay

**Initialization:**
- `init_from`: 'scratch', 'resume', or 'gpt2*' (pretrained)
- `eval_only`: Run evaluation only (no training)

## Usage Patterns

### 1. Quick Experimentation
```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

### 2. Production Training with Custom Overrides
```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py --batch_size=16 --learning_rate=5e-4
```

### 3. Fine-tuning Workflow
```bash
# Fine-tune on custom dataset
python train.py config/finetune_shakespeare.py --dataset=my_custom_dataset --max_iters=50
```

### 4. Benchmarking Pretrained Models
```bash
# Evaluate all GPT-2 variants
for config in config/eval_gpt2*.py; do
    python train.py $config
done
```

## Relationship with Other Components

### Dependencies
- **train.py**: Primary consumer of these configurations
- **configurator.py**: Configuration loading mechanism
- **data/**: Dataset directories referenced by `dataset` parameter

### Configuration Loading Order
1. `train.py` defines default parameters as global variables
2. `exec(open('configurator.py').read())` loads configuration system
3. Configuration file (if provided) overrides defaults: `exec(open(config_file).read())`
4. Command-line arguments override configuration file: `--key=value`

### Dataset Requirements
Configuration files reference datasets that must exist in `data/` with:
- `train.bin`, `val.bin`: Preprocessed binary data
- `meta.pkl`: Vocabulary and tokenizer metadata

## Design Philosophy

**Simplicity over Complexity**
- No YAML/JSON parsing or complex configuration frameworks
- Pure Python files with natural syntax
- Direct variable assignment (e.g., `learning_rate = 1e-3`)
- Easy to understand and modify

**Composability**
- Base configurations can be overridden via command line
- Multiple configuration files can be chained
- Type-safe override checking in `configurator.py`

**Reproducibility**
- Each configuration documents expected hardware and outcomes
- Training configs include estimated training time
- Wandb integration for experiment tracking

## Tips for Creating Custom Configurations

1. **Start from existing config**: Copy `train_shakespeare_char.py` for small experiments
2. **Scale batch size carefully**: Adjust `gradient_accumulation_steps` to maintain effective batch size
3. **Match learning rate to model size**: Smaller models can use higher LR (1e-3), larger models need lower LR (3e-4 to 6e-4)
4. **Set decay_iters = max_iters**: For proper cosine decay schedule
5. **Use eval_interval wisely**: Frequent evals slow training but help catch issues early
6. **Fine-tuning guidelines**: Use pretrained init, constant/low LR, fewer iterations

## Example: Creating a Custom Config

```python
# config/train_my_model.py
# Train medium-sized GPT on custom dataset

out_dir = 'out-my-model'
wandb_log = True
wandb_project = 'my-project'
wandb_run_name = 'gpt-medium-custom'

dataset = 'my_dataset'
batch_size = 8
block_size = 512
gradient_accumulation_steps = 16

# Medium model
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1

learning_rate = 5e-4
max_iters = 50000
lr_decay_iters = 50000
warmup_iters = 1000

eval_interval = 500
```

Then run: `python train.py config/train_my_model.py`
