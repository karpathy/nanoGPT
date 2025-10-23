# Shakespeare Dataset Preparation (Word-Level)

## Overview
This directory contains the preparation script for the Tiny Shakespeare dataset using word-level (subword) tokenization via GPT-2 BPE. Ideal for fine-tuning pretrained GPT-2 models.

## Purpose
Download and tokenize the Tiny Shakespeare corpus using GPT-2 BPE tokenization, making it compatible with pretrained GPT-2 models for transfer learning and fine-tuning experiments.

## Key File

### `prepare.py`
**Dataset download and tokenization script**

**Dataset Source:**
- URL: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Size: ~1.1MB text file
- Content: Concatenated works of Shakespeare

**Processing Pipeline:**
1. Download `input.txt` if not present (auto-download on first run)
2. Split into train/val (90% / 10%)
3. Tokenize using GPT-2 BPE via tiktoken
4. Export to binary files (numpy uint16 arrays)

**Output Files:**
- `train.bin`: 301,966 tokens (~604KB)
- `val.bin`: 36,059 tokens (~72KB)
- `input.txt`: Raw Shakespeare text (~1.1MB)

**Tokenization Details:**
- Encoder: GPT-2 BPE (tiktoken)
- Vocabulary: 50,257 tokens (shared with GPT-2)
- Method: `enc.encode_ordinary()` (standard BPE encoding)
- Format: uint16 numpy array

**Usage:**
```bash
cd data/shakespeare
python prepare.py
```

**Runtime:**
- Download: < 1 second (small file)
- Tokenization: < 5 seconds
- Total: Very fast, suitable for quick experiments

## Use Cases

### 1. Fine-tuning Pretrained Models
**Primary use case** - Transfer learning from GPT-2:
```bash
# Fine-tune GPT-2 XL on Shakespeare
python train.py config/finetune_shakespeare.py
```

**Benefits:**
- Compatible with pretrained GPT-2 weights (shared tokenizer)
- Fast fine-tuning (only ~20 iterations needed)
- Demonstrates transfer learning effectiveness

### 2. Quick Evaluation
Test pretrained models on out-of-distribution text:
```bash
python train.py config/finetune_shakespeare.py --eval_only=True --init_from=gpt2
```

### 3. Style Transfer Experiments
Train models to generate Shakespearean text by fine-tuning modern language models.

## Comparison with Character-Level Version

**Word-Level (`shakespeare/`):**
- Tokenization: GPT-2 BPE (subword)
- Vocabulary: 50,257 tokens
- Tokens: ~302K training tokens
- Use case: Fine-tuning pretrained models
- Advantage: Leverages pretrained weights

**Character-Level (`shakespeare_char/`):**
- Tokenization: Character mapping
- Vocabulary: 65 characters
- Tokens: ~1M training characters
- Use case: Training from scratch
- Advantage: Simpler, no external dependencies

**Key Difference:**
Word-level uses GPT-2's BPE tokenizer, making it directly compatible with pretrained models. Character-level uses simple character-to-integer mapping, better for understanding from-scratch training.

## Dependencies

**Required Libraries:**
- `numpy`: Binary file operations
- `tiktoken`: GPT-2 BPE tokenization
- `requests`: Dataset download

**Installation:**
```bash
pip install numpy tiktoken requests
```

## Data Statistics

**Raw Data:**
- Total characters: 1,115,394
- Unique characters: 65
- Documents: Concatenated Shakespeare works

**Tokenized Data (BPE):**
- Train split: 301,966 tokens (90%)
- Val split: 36,059 tokens (10%)
- Average compression: ~3.7 characters per token

**Split Strategy:**
- Simple 90/10 split at character level
- No shuffling (maintains text continuity)
- Deterministic split (always same result)

## Relationship with Training Code

### Configuration Files
**Primary config**: `config/finetune_shakespeare.py`
```python
dataset = 'shakespeare'
init_from = 'gpt2-xl'  # Start from pretrained
learning_rate = 3e-5   # Low LR for fine-tuning
max_iters = 20         # Quick fine-tuning
```

### Integration with `train.py`
```python
# train.py automatically loads
data_dir = 'data/shakespeare'
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
```

### Sample Generation
After fine-tuning, generate Shakespearean text:
```bash
python sample.py --init_from=resume --out_dir=out-shakespeare --start="To be or not to be"
```

## Performance Characteristics

**Training Speed:**
- Small dataset allows very frequent evaluation
- 1 epoch ≈ 9.2 iterations (with config defaults)
- Full fine-tuning: ~5-10 minutes on single GPU

**Memory Requirements:**
- Minimal: < 1GB disk space
- Fits entirely in memory
- No memmap overhead needed (but used for consistency)

**Overfitting Risk:**
- Small dataset (302K tokens) overfits quickly
- Use dropout during fine-tuning
- Monitor validation loss closely
- Save only when validation improves

## Best Practices

### Fine-tuning Tips
1. **Start from large model**: GPT-2 XL (1558M) works best
2. **Use low learning rate**: 1e-5 to 5e-5 range
3. **Short training**: 10-50 iterations usually sufficient
4. **No LR decay**: Constant learning rate for fine-tuning
5. **Small batch size**: Effective batch size ~32K tokens

### Evaluation Strategy
```python
eval_interval = 5      # Evaluate every 5 iterations
eval_iters = 40        # 40 batches for validation
always_save_checkpoint = False  # Only save improvements
```

### Avoiding Overfitting
- Monitor train/val loss gap
- Stop when validation loss increases
- Use dropout (0.1-0.2)
- Consider data augmentation (varies by application)

## Tokenization Compatibility

**GPT-2 BPE Advantages:**
- Direct weight transfer from pretrained models
- Efficient encoding (subword units)
- Handles unseen words via BPE composition
- Standard vocabulary (50,257 tokens)

**Usage in Code:**
```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")

# Encoding
text = "To be or not to be"
tokens = enc.encode_ordinary(text)

# Decoding
decoded = enc.decode(tokens)
```

**No `meta.pkl` needed** - GPT-2 tokenizer is standardized and loaded directly from tiktoken.

## Troubleshooting

**Issue: Download fails**
- Check internet connection
- Verify URL is accessible: `curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Manual download: Save URL content to `input.txt`

**Issue: Import errors**
- Install tiktoken: `pip install tiktoken`
- Check numpy version: `pip install numpy>=1.20`

**Issue: Fine-tuning doesn't improve**
- Try lower learning rate (1e-5)
- Increase gradient accumulation steps
- Train for more iterations (50-100)
- Check validation loss is actually evaluated

## Educational Value

**Good for learning:**
- Transfer learning concepts
- Fine-tuning vs training from scratch
- Tokenization impact on model performance
- Overfitting detection and prevention
- Small-scale experimentation (fast iteration)

**Comparison experiments:**
- Fine-tune vs train from scratch
- Different GPT-2 sizes (base, medium, large, xl)
- Different learning rates
- With/without dropout

## Notes

- Dataset is intentionally small for educational purposes
- Not suitable for production Shakespeare generation without augmentation
- Demonstrates transfer learning effectiveness (pretrained model adapts quickly)
- Alternative to character-level version for pretrained model experiments
