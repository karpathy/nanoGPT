# textfile_char

Character-level dataset preparation for nanoGPT.

## Usage

1. Place your `.txt` file(s) in this directory
2. Run `python prepare.py`

## What it does

- Scans this directory (recursively) for all `.txt` files
- Combines them into a single dataset
- Builds a character-level vocabulary from all unique characters
- Splits data into 90% train / 10% validation
- Saves encoded data as `train.bin` and `val.bin` (uint16)
- Saves vocabulary mappings (`stoi`, `itos`) in `meta.pkl`

## Output files

| File | Description |
|------|-------------|
| `train.bin` | Training tokens (90% of data) |
| `val.bin` | Validation tokens (10% of data) |
| `meta.pkl` | Vocabulary size + encoder/decoder mappings |
