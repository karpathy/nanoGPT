# LOZO with nanoGPT

This implementation integrates Low-Rank Zero-Order (LOZO) optimization with nanoGPT for pretraining language models without traditional backpropagation-based gradient computation.

## What is LOZO?

LOZO (Low-Rank Zero-Order) is an optimization technique that uses zeroth-order methods with low-rank perturbations for training neural networks. Unlike traditional methods that require backpropagation, LOZO estimates gradients by evaluating the model at slightly perturbed parameter values. The key benefits are:

1. Memory efficiency - No need to store intermediate activations for backpropagation
2. Low-rank perturbations - More efficient gradient estimation compared to standard zeroth-order methods
3. Works with non-differentiable objectives - Can optimize metrics that aren't differentiable

## How to use 

### Quick Start with Shakespeare Dataset

For testing on a smaller dataset like Shakespeare:

```sh
# First, prepare the Shakespeare dataset
python data/shakespeare/prepare.py

# Train using LOZO on Shakespeare dataset
python lozo_train.py config/train_lozo_shakespeare.py
```

This will train a smaller model on the Shakespeare dataset using LOZO optimization, which is great for quick testing.

### Pretraining GPT-2 with LOZO

For pretraining on a larger dataset like OpenWebText:

```sh
# First, prepare the OpenWebText dataset
python data/openwebtext/prepare.py

# Train using LOZO
python lozo_train.py config/train_lozo_gpt2.py
```

For multi-GPU training with DDP:

```sh
torchrun --standalone --nproc_per_node=8 lozo_train.py config/train_lozo_gpt2.py
```

## Configuration Options

The LOZO implementation adds three key parameters to the standard nanoGPT configuration:

- `zo_eps`: Perturbation size for zero-order gradient estimation (default: 1e-3)
- `rank_r`: Rank for low-rank perturbation matrices (default: 4)
- `step_interval`: Number of steps between V matrix updates (default: 10)

This `step_interval` parameter controls how often the V matrices are resampled (every ν steps in the LOZO paper). This is a crucial parameter that affects the stability and efficiency of the optimization.

You can modify these parameters in the config files to tune the LOZO optimization process.

## LOZO vs. Standard Training

LOZO offers several advantages:

1. **Memory Efficiency**: LOZO doesn't need to store intermediate activations for backpropagation, reducing memory usage.
2. **Low-Rank Updates**: The perturbations and updates use low-rank matrices, making the optimization process more efficient.
3. **Non-Differentiable Objectives**: LOZO can optimize metrics that aren't differentiable.

However, it typically requires more forward passes to achieve similar performance compared to standard training.

## Implementation Details

The implementation is based on the paper "Enhancing Zeroth-order Fine-tuning for Language Models with Low-rank Structures" and uses:

1. Low-rank perturbations to model parameters
2. Gradient estimation via zeroth-order optimization
3. Parameter updates that respect the low-rank structure
4. Periodic update of V matrices according to the step_interval

The code is heavily commented to explain the LOZO-specific components.

## Tips for Best Results

- **Adjust `zo_eps`**: This controls the perturbation magnitude. If training is unstable, try smaller values.
- **Tune `rank_r`**: Higher ranks can capture more complex gradient structure but use more memory.
- **Set `step_interval`**: Smaller values (more frequent updates) can improve exploration but may be less stable. Larger values can improve stability.
- **Training Time**: LOZO typically requires more iterations than standard training, so be patient. 