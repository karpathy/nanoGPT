# Attention Sink Experiment

This experiment implements and tests the **Attention Sink** mechanism described in the paper ["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453) by Xiao et al.

## Background

The attention sink paper makes an important observation: language models tend to allocate significant attention to initial tokens (especially the first token), even when those tokens are not semantically important. These initial tokens act as "attention sinks" that absorb attention weights.

### Key Findings from the Paper:

1. **Attention Sink Phenomenon**: Models consistently pay high attention to the first few tokens, regardless of their semantic content
2. **Streaming Inference**: By preserving these "sink" tokens plus a sliding window of recent tokens, models can perform efficient streaming inference on very long sequences
3. **Performance**: Attention sink + sliding window maintains similar performance to full attention while being much more memory efficient

## Implementation

### Model Architecture Changes

We've implemented `AttentionSinkCausalSelfAttention` in `model.py`, which modifies the standard causal attention mechanism:

**Standard Attention:**
- Each position can attend to all previous positions (full causal attention)
- Memory grows quadratically with sequence length

**Attention Sink:**
- Each position can attend to:
  1. First `sink_size` tokens (attention sinks) - always visible
  2. A sliding window of `window_size` most recent tokens
- Memory is bounded: O(T × (sink_size + window_size))

### Configuration Parameters

Added to `GPTConfig`:
- `use_attention_sink`: Enable/disable attention sink mechanism (default: False)
- `sink_size`: Number of initial tokens to keep as attention sinks (default: 4)
- `window_size`: Size of sliding attention window (default: 252)

## Running the Experiment

### 1. Prepare Data

```bash
# Prepare Shakespeare character-level dataset
python data/shakespeare_char/prepare.py
```

### 2. Train Baseline Model

Train a standard attention model for comparison:

```bash
python train.py config/train_baseline_for_sink.py
```

This will:
- Train a 6-layer GPT model with standard causal attention
- Save checkpoints to `out-baseline-sink/`
- Log to W&B project `attention-sink-experiment`

### 3. Train Attention Sink Model

Train the attention sink variant:

```bash
python train.py config/train_attention_sink.py
```

This will:
- Train a 6-layer GPT model with attention sink mechanism
- Use 4 sink tokens + 252 token sliding window
- Save checkpoints to `out-attention-sink/`
- Log to same W&B project for comparison

### 4. Evaluate and Compare

Compare the two models:

```bash
python eval_attention_sink.py \
    --baseline_checkpoint out-baseline-sink/ckpt.pt \
    --sink_checkpoint out-attention-sink/ckpt.pt \
    --data_dir data/shakespeare_char
```

This will:
- Evaluate perplexity on validation set
- Generate text samples from both models
- Compare model parameters
- Save detailed results to `attention_sink_results.txt`

### 5. Visualize Attention Patterns

Analyze and visualize attention patterns:

```bash
python visualize_attention.py \
    --standard_checkpoint out-baseline-sink/ckpt.pt \
    --sink_checkpoint out-attention-sink/ckpt.pt \
    --output_dir attention_analysis
```

This will generate:
- `attention_standard.png`: Heatmap of standard attention patterns
- `attention_sink.png`: Heatmap showing sink + window pattern
- `attention_comparison.png`: Statistical comparison of attention distributions

**Note**: Attention extraction only works when NOT using Flash Attention. For visualization, you may need to disable Flash Attention or use PyTorch < 2.0.

## Expected Results

### What to Look For:

1. **Perplexity**: Attention sink model should achieve similar perplexity to baseline
   - Small increase in perplexity is expected due to limited context
   - Difference should be < 5% for effective window sizes

2. **Attention Patterns**:
   - Standard model: Smooth triangular attention pattern (full causal)
   - Attention sink: Distinct pattern with:
     - Vertical stripe for sink tokens (always attended)
     - Diagonal band for sliding window
     - Clear boundaries at sink_size

3. **Memory Efficiency**:
   - Standard: O(T²) attention computation
   - Attention Sink: O(T × (sink_size + window_size))
   - For long sequences, this is a huge saving

4. **Generation Quality**:
   - Both models should produce coherent Shakespeare-style text
   - Attention sink may struggle with very long-range dependencies
   - For typical contexts (< block_size), quality should be comparable

## Configuration Options

### For Quick Testing (CPU/Small GPU):

```python
# In config file
batch_size = 32
block_size = 128
n_layer = 4
n_head = 4
n_embd = 256
max_iters = 2000
device = 'cpu'  # or 'cuda'
compile = False
```

### For Full Experiment (Large GPU):

```python
batch_size = 64
block_size = 512
n_layer = 12
n_head = 12
n_embd = 768
max_iters = 10000
sink_size = 4
window_size = 508  # 512 - 4 = 508
```

## Files Added

1. **Model Implementation**:
   - `model.py`: Added `AttentionSinkCausalSelfAttention` class and config parameters

2. **Training Configs**:
   - `config/train_baseline_for_sink.py`: Baseline model config
   - `config/train_attention_sink.py`: Attention sink model config

3. **Evaluation Scripts**:
   - `eval_attention_sink.py`: Compare models and evaluate metrics
   - `visualize_attention.py`: Visualize and analyze attention patterns

4. **Documentation**:
   - `ATTENTION_SINK_EXPERIMENT.md`: This file

## Interpreting Results

### Attention Heatmaps:

The attention heatmap shows attention weights where:
- **X-axis**: Key positions (attended to)
- **Y-axis**: Query positions (attending from)
- **Color**: Attention weight (bright = high attention)

**Standard Attention:**
- Lower triangular matrix (causal masking)
- Smooth gradient of attention over all previous tokens

**Attention Sink:**
- Vertical stripe at x = 0 to sink_size (sink tokens always visible)
- Diagonal band showing sliding window
- Black regions where attention is masked out

### Statistical Metrics:

1. **Sink Attention**: % of attention allocated to sink tokens
   - Should be higher in attention sink model
   - Validates the attention sink phenomenon

2. **First Token Attention**: Attention to the very first token
   - Often surprisingly high (30-40%)
   - Core observation from the paper

3. **Attention Entropy**: How distributed vs focused is attention
   - Higher entropy = more distributed
   - Lower entropy = more focused

4. **Attention Span**: Average distance between query and attended keys
   - Bounded by window_size in attention sink model
   - Unbounded in standard model

## Extensions and Future Work

1. **Longer Sequences**: Test on sequences longer than block_size to see streaming benefits
2. **Different Window Sizes**: Ablation study on sink_size and window_size
3. **Other Datasets**: Test on different domains (code, books, etc.)
4. **Positional Encodings**: Investigate how RoPE or ALiBi interact with attention sinks
5. **KV Cache Management**: Implement efficient KV caching for streaming inference

## References

- Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). **Efficient Streaming Language Models with Attention Sinks**. arXiv:2309.17453
- Radford, A., et al. (2019). **Language Models are Unsupervised Multitask Learners**. OpenAI Blog.

## Citation

If you use this implementation in your research, please cite the original attention sink paper:

```bibtex
@article{xiao2023attention,
  title={Efficient Streaming Language Models with Attention Sinks},
  author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
  journal={arXiv preprint arXiv:2309.17453},
  year={2023}
}
```
