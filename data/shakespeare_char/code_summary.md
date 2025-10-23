# Shakespeare Dataset Preparation (Character-Level)

## Overview
This directory contains the preparation script for the Tiny Shakespeare dataset using character-level tokenization. Ideal for training small models from scratch and learning transformer fundamentals.

## Purpose
Download and process the Tiny Shakespeare corpus with simple character-to-integer mapping, creating a minimal dataset perfect for educational purposes, debugging, and running on laptops/consumer hardware.

## Key File

### `prepare.py`
**Dataset download and character-level tokenization script**

**Dataset Source:**
- URL: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Size: ~1.1MB text file
- Content: Concatenated works of Shakespeare (1,115,394 characters)

**Processing Pipeline:**
1. Download `input.txt` if not present (auto-download on first run)
2. Extract unique characters from text (vocabulary discovery)
3. Create bidirectional character-to-integer mappings (encode/decode)
4. Split into train/val (90% / 10%)
5. Encode text as integer sequences
6. Export to binary files and metadata pickle

**Output Files:**
- `train.bin`: 1,003,854 tokens (~2MB)
- `val.bin`: 111,540 tokens (~223KB)
- `meta.pkl`: Vocabulary and encoder/decoder functions
- `input.txt`: Raw Shakespeare text (~1.1MB)

**Vocabulary:**
- Size: 65 characters
- Characters: ` !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`
- Mapping: Sequential integers 0-64

**Usage:**
```bash
cd data/shakespeare_char
python prepare.py
```

**Runtime:**
- Download: < 1 second
- Processing: < 1 second (simple character mapping)
- Total: Extremely fast

## Character-Level Tokenization

### Encoding/Decoding
**Simple character mapping:**
```python
# Character to index
stoi = {'a': 0, 'b': 1, ...}

# Index to character
itos = {0: 'a', 1: 'b', ...}

# Encode function
def encode(s):
    return [stoi[c] for c in s]

# Decode function
def decode(l):
    return ''.join([itos[i] for i in l])
```

### Metadata File (`meta.pkl`)
```python
meta = {
    'vocab_size': 65,
    'itos': {0: 'a', 1: 'b', ...},  # Index to string
    'stoi': {'a': 0, 'b': 1, ...},  # String to index
}
```

**Used by `train.py` and `sample.py`:**
- Model uses `vocab_size` for embedding layer size
- Sampling script uses `encode`/`decode` for text generation
- Automatically loaded from `data/shakespeare_char/meta.pkl`

## Use Cases

### 1. Training from Scratch
**Primary use case** - Learn transformer training fundamentals:
```bash
python train.py config/train_shakespeare_char.py
```

**Configuration (`config/train_shakespeare_char.py`):**
- Model: "Baby GPT" (6 layers, 6 heads, 384 dim)
- Context: 256 characters
- Batch size: 64
- Iterations: 5000 (~15 minutes)
- Learning rate: 1e-3
- Dropout: 0.2

**Benefits:**
- Fast iteration cycles
- Runs on laptops/CPUs
- Easy to understand and debug
- Complete training in < 20 minutes

### 2. Architecture Experiments
Test model architecture changes quickly:
- Different layer counts
- Attention head configurations
- Embedding dimensions
- Context window sizes

### 3. Educational Demonstrations
Perfect for teaching:
- Character-level language modeling
- Transformer architecture
- Training dynamics (loss curves, overfitting)
- Sampling strategies (temperature, top-k)

### 4. Debugging
Small dataset makes debugging easy:
- Fast forward/backward passes
- Quick convergence
- Easy to spot errors
- Minimal resource requirements

## Comparison with Word-Level Version

**Character-Level (`shakespeare_char/`):**
- ✅ Simple implementation (no external tokenizer)
- ✅ Self-contained vocabulary discovery
- ✅ Train from scratch (no pretrained weights)
- ✅ Smaller model sizes (65 vocab vs 50K+)
- ✅ Runs on CPU/laptop
- ❌ Cannot use pretrained GPT-2 weights
- ❌ Longer sequences (1M chars vs 300K tokens)

**Word-Level (`shakespeare/`):**
- ✅ Compatible with pretrained GPT-2
- ✅ Efficient encoding (subword units)
- ✅ Fine-tuning transfer learning
- ❌ Requires tiktoken library
- ❌ Large vocabulary (50,257 tokens)
- ❌ Needs GPU for pretrained models

**When to use Character-Level:**
- Learning transformer fundamentals
- Limited compute resources
- Want to train from scratch
- Prefer simplicity over performance

**When to use Word-Level:**
- Fine-tuning pretrained models
- Need better generation quality
- Have GPU available
- Want faster convergence

## Dependencies

**Required Libraries:**
- `numpy`: Binary file operations
- `pickle`: Metadata serialization
- `requests`: Dataset download

**Installation:**
```bash
pip install numpy requests
```

**No tokenizer dependencies** - Uses built-in Python string operations.

## Data Statistics

**Raw Data:**
- Total characters: 1,115,394
- Unique characters: 65
- Character set: English letters, punctuation, digits
- Source: Concatenated Shakespeare works

**Tokenized Data:**
- Train: 1,003,854 characters (90%)
- Val: 111,540 characters (10%)
- Vocabulary size: 65
- Token IDs: 0-64 (stored as uint16)

**Split Strategy:**
- Simple 90/10 split at character position
- No shuffling (preserves text continuity)
- Deterministic (reproducible)

## Relationship with Training Code

### Configuration File
**Primary config**: `config/train_shakespeare_char.py`

Key settings:
```python
dataset = 'shakespeare_char'
out_dir = 'out-shakespeare-char'

# Baby GPT architecture
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256  # Character context window

# Training
batch_size = 64
max_iters = 5000
learning_rate = 1e-3
dropout = 0.2

# Checkpointing
always_save_checkpoint = False  # Only save when val improves
```

### Model Initialization
```python
# train.py loads vocabulary from meta.pkl
meta_path = 'data/shakespeare_char/meta.pkl'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']  # 65

# Create model with small vocab
model = GPT(GPTConfig(vocab_size=65, ...))
```

### Text Generation
```python
# sample.py uses custom encode/decode
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Generate text
prompt = "ROMEO:"
tokens = encode(prompt)
generated = model.generate(tokens, max_new_tokens=500)
text = decode(generated)
```

## Training Characteristics

### Expected Results
**After 5000 iterations (~15 minutes):**
- Train loss: ~1.0-1.2
- Val loss: ~1.4-1.6
- Quality: Recognizable Shakespeare-like text
- Overfitting: Mild (train/val gap ~0.3-0.4)

### Performance
**On consumer hardware:**
- CPU (MacBook): ~2 seconds/iteration → 3 hours total
- GPU (RTX 3060): ~50ms/iteration → 4 minutes total
- GPU (A100): ~20ms/iteration → 2 minutes total

### Memory Requirements
- Model: ~10M parameters → ~40MB
- Dataset: ~2MB on disk, fits entirely in RAM
- Training: < 2GB total memory
- **Can run on 8GB laptop**

## Best Practices

### Training Tips
1. **Use dropout**: 0.2 helps prevent overfitting on small dataset
2. **Monitor validation loss**: Stop when gap with train loss grows
3. **Save selectively**: `always_save_checkpoint=False` saves best only
4. **Frequent evaluation**: `eval_interval=250` catches overfitting early

### Generation Tips
```bash
# Generate samples after training
python sample.py \
  --out_dir=out-shakespeare-char \
  --start="ROMEO:" \
  --num_samples=5 \
  --temperature=0.8 \
  --top_k=200
```

**Temperature effects:**
- 0.8: Balanced, coherent text
- 1.0: More diverse, some nonsense
- 0.5: Conservative, repetitive

### Experimentation Ideas
1. **Vary model size**: Try 4/8/12 layers
2. **Context length**: 128/256/512 characters
3. **Learning rate**: Sweep 5e-4 to 5e-3
4. **Dropout**: 0.0/0.1/0.2/0.3
5. **Batch size**: 32/64/128

## Troubleshooting

**Issue: Out of vocabulary characters**
- Dataset contains only 65 unique characters
- Custom text must use same character set
- Solution: Filter input or retrain with expanded vocab

**Issue: Model not improving**
- Check learning rate (try 1e-3)
- Verify data loading (inspect batches)
- Ensure sufficient capacity (6 layers minimum)

**Issue: Overfitting quickly**
- Increase dropout (0.3)
- Reduce model size (4 layers, 256 dim)
- Add weight decay
- Early stopping

**Issue: Generation quality poor**
- Train longer (10K iterations)
- Increase model size (8 layers, 512 dim)
- Use lower temperature (0.7)
- Sample with top-k=100

## Educational Value

### Perfect for Learning
1. **Minimal complexity**: No BPE, no external tokenizers
2. **Fast iteration**: Results in minutes
3. **Observable overfitting**: Small dataset shows train/val gap
4. **Complete pipeline**: Data prep → training → generation
5. **Debugging friendly**: Easy to inspect tokens and outputs

### Key Lessons
- Character-level modeling basics
- Vocabulary creation and management
- Overfitting detection and prevention
- Sampling strategies and temperature effects
- Training hyperparameter tuning

### Comparison Experiments
- Character-level vs word-level (compare with `shakespeare/`)
- From-scratch vs fine-tuning
- Small model vs large model
- Different dropout rates
- Various context lengths

## Advanced Usage

### Custom Vocabulary
Modify `prepare.py` to include more characters:
```python
# Add numbers, special chars
chars = sorted(list(set(data + "0123456789@#")))
```

### Data Augmentation
- Concatenate multiple text sources
- Apply simple text transformations
- Mix with modern English for comparison

### Model Scaling
```python
# Larger model (still trainable on laptop)
n_layer = 8
n_head = 8
n_embd = 512
# ~35M parameters, ~30 minutes training
```

## Notes

- Intentionally simple for educational purposes
- Character-level is less efficient than BPE but easier to understand
- Good starting point before tackling larger datasets
- Demonstrates transformer capabilities on minimal data
- Perfect for experimenting with architecture changes
- Can achieve decent Shakespeare-like text generation with small model
- Trade-off: Simplicity vs generation quality
