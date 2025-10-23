# OpenWebText Dataset Preparation

## Overview
This directory contains the data preparation script for the OpenWebText dataset, a large-scale web text corpus used for training GPT-2 scale models.

## Purpose
Download, process, and tokenize the OpenWebText dataset into memory-mapped binary files optimized for efficient training with minimal I/O overhead.

## Key File

### `prepare.py`
**Dataset preparation and tokenization script**

**Dataset Source:**
- HuggingFace dataset: `openwebtext`
- Size: ~54GB (cached), 8M+ documents (8,013,769)
- Content: Web text from Reddit links with high karma

**Processing Pipeline:**
1. Download dataset via HuggingFace `datasets` library
2. Create train/val split (99.95% / 0.05%)
   - Train: 8,009,762 documents
   - Val: 4,007 documents
3. Tokenize using GPT-2 BPE (tiktoken)
   - Add end-of-text token (50256) after each document
4. Concatenate all tokens into memory-mapped binary files
5. Write as numpy uint16 arrays (GPT-2 vocab < 2^16)

**Output Files:**
- `train.bin`: ~17GB, ~9B tokens (9,035,582,198)
- `val.bin`: ~8.5MB, ~4M tokens (4,434,897)

**Configuration:**
- `num_proc`: Number of CPU workers for tokenization (default: 8)
- `num_proc_load_dataset`: Workers for dataset loading (default: 8)
- Adjust based on CPU cores and network speed

**Usage:**
```bash
cd data/openwebtext
python prepare.py
```

**Runtime:**
- Download: Varies by network speed (~54GB)
- Tokenization: ~30-60 minutes (8 workers)
- Total disk space: ~54GB (cache) + ~17GB (processed) = ~71GB

**Tokenization Details:**
- Encoder: GPT-2 BPE via tiktoken
- Vocabulary size: 50,257 tokens
- Special tokens: End-of-text (EOT) token appended to each document
- Encoding function: `enc.encode_ordinary()` (ignores special tokens during encoding)
- Token IDs stored as uint16 (2 bytes per token)

**Memory-Mapped Format:**
Binary files use numpy memmap for zero-copy loading:
```python
# Read the data later
import numpy as np
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
```

## Dependencies

**Required Libraries:**
- `numpy`: Array operations and memmap
- `tiktoken`: GPT-2 BPE tokenization
- `datasets`: HuggingFace datasets library
- `tqdm`: Progress bars
- `requests`: (transitive dependency)

**Installation:**
```bash
pip install numpy tiktoken datasets tqdm
```

## Dataset Split Strategy

**Train/Val Split:**
- Split ratio: 0.0005 test size (0.05% validation)
- Seed: 2357 (for reproducibility)
- Shuffle: True (random split)

**Rationale:**
- Large dataset allows tiny validation set
- Validation set still has 4M tokens (sufficient for loss estimation)
- Maximizes training data usage

## Relationship with Training Code

**Integration with `train.py`:**
```python
# In train.py, load memmap files
data_dir = 'data/openwebtext'
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
```

**Memory Efficiency:**
- Memmap files are not loaded into RAM entirely
- OS handles paging automatically
- Minimal memory footprint during training
- Zero-copy access to disk data

**Training Configuration:**
- Default dataset in `train.py`: `dataset = 'openwebtext'`
- Config file: `config/train_gpt2.py` targets this dataset
- Expected training tokens: 300B (requires ~33 epochs over 9B tokens)

## Performance Considerations

**Parallelization:**
- Tokenization is CPU-bound (use multiple workers)
- Batch writing reduces I/O overhead (1024 batches)
- Sharding improves parallel performance

**Storage Format:**
- uint16 saves 50% space vs uint32
- Contiguous memory layout for fast sequential access
- Memory-mapped files reduce RAM requirements during training

**Optimization:**
```python
# Adjust workers based on your system
num_proc = os.cpu_count() // 2  # Conservative estimate
```

## Troubleshooting

**Issue: Out of disk space**
- Ensure ~75GB free space (54GB cache + 17GB processed)
- Cache location: `~/.cache/huggingface/datasets/`

**Issue: Slow download**
- Dataset is 54GB, download time depends on internet speed
- Consider running overnight or on high-bandwidth connection

**Issue: Out of memory during tokenization**
- Reduce `num_proc` to lower parallel workers
- Increase system RAM or use swap space

**Issue: Tokenization slow**
- Increase `num_proc` if CPU cores available
- Check CPU usage to ensure workers are busy

## Notes

**Dataset Quality:**
- OpenWebText is a recreation of OpenAI's WebText dataset
- Scraped from Reddit links with karma >= 3
- Generally high-quality, diverse web text
- Suitable for general-purpose language modeling

**GPT-2 Compatibility:**
- Uses identical tokenization to OpenAI GPT-2
- Vocab size: 50,257 (padded to 50,304 in model for efficiency)
- Can load pretrained GPT-2 weights and continue training

**Alternative Datasets:**
- See `data/shakespeare/` for small-scale experimentation
- See `data/shakespeare_char/` for character-level modeling
- Custom datasets should follow same binary format
