# Tokenization

This folder is a template data folder, with a script provides a utility for
tokenizing text data using alternative methods.

Currently SentencePiece, TikToken, and character-level tokenizations are
supported, with more tokenization method support planned.

## Currently Supported Tokenizations

- **SentencePiece Tokenization**
- **TikToken Tokenization**
- **Character-Level Tokenization**

## Usage

### Prerequisites

Ensure you have Python installed on your system along with the necessary libraries: `numpy`, `pickle`, `sentencepiece`, and `tiktoken`.

### Command Line Arguments

- `input_file`: Path to the input text file.
- `--method`: Tokenization method (`sentencepiece`, `tiktoken`, `char`). Default is `sentencepiece`.
- `--vocab_size`: Vocabulary size for the SentencePiece model. Default is 500.

#### 1. Create a New Data Folder

First copy this folder into a new folder:

```bash
# from the ./data directory
cp -r ./template ./new_data_folder
```

#### 2. Add data to folder

Obtain a text format of the data for training.

### 3. Run tokenization script

This script takes in a text file as its argument for tokenization.

Afterward it produces the train.bin and val.bin (and meta.pkl if not tiktoken)
then you'll be able to begin training with the train.py script.

#### SentencePiece

```bash
python3 text_tokenizer.py input.txt --method sentencepiece --vocab_size 1000
```

#### TikToken

```bash
python3 text_tokenizer.py input.txt --method tiktoken
```
#### Character Level Tokenization

This command will tokenize the text in from the input file at the character level.

```bash
python3 text_tokenizer.py input.txt --method char
```

## Relevant Resources and References

This section provides links to research papers and GitHub repositories related to the tokenization methods used in this script. These resources can offer deeper insights into the algorithms and their implementations.

### SentencePiece

- [Read the Paper](https://arxiv.org/abs/1808.06226)
- [SentencePiece Github Repository](https://github.com/google/sentencepiece)

### TikToken

- [General Usage Guide](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)
- [TikToken Github Repository](https://github.com/openai/tiktoken)

## Open to Contributions

- [ ] Add feature to take in a file with set of multi-character tokens for custom tokenization (e.g. char level tokenization but custom word-level tokenization list)
- [ ] Add byte-level tokenization options
- [ ] Add argparse arguments for more features of SentencePiece and TikToken
