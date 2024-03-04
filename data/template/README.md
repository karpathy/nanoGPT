# Tokenization

This folder is a template data folder, with a script provides a utility for
tokenizing text data using alternative methods.

Currently SentencePiece, TikToken, and character-level tokenizations are
supported, with more tokenization method support planned.

Additionally, phonemization using espeak via a shell script is supported to
preprocess text data into phoneme representations.

## Currently Supported Tokenizations

- **SentencePiece Tokenization**
- **TikToken Tokenization**
- **Character-Level Tokenization**

## Usage

### Prerequisites

Ensure you have Python installed on your system along with the necessary
libraries: `numpy`, `pickle`, `sentencepiece`, and `tiktoken`.

Also for phonemization ensure that `espeak` and `GNU Parallel` are installed.

##### 1. Create a New Data Folder

First copy this folder into a new folder:

```bash
# from the ./data directory
cp -r ./template ./new_data_folder
```

##### 2. Add data to folder

Obtain a text format of the data for training.

Note: make sure not to check in only the scripts and instructions, not the dataset.

### 3. Run tokenization script

Finally, run `prepare.py` script to process the dataset for training.

#### Examples:

##### SentencePiece

```bash
python3 prepare.py -t input.txt --method sentencepiece --vocab_size 1000
```

##### TikToken

```bash
python3 prepare.py -t input.txt --method tiktoken
```

##### Character Level Tokenization

This command will tokenize the text in from the input file at the character level.

```bash
python3 prepare.py -t input.txt --method char
```

##### Custom

```bash
python3 prepare.py -t input.txt --method custom --tokens_file phoneme_list.txt
```

### Additional details about the `prepare.py` script

#### `prepare.py` Command Line Arguments

This script takes in a text file as its argument for tokenization, and
additional parameters for tokenization strategy and their parameters.

- `input_file`: Path to the input text file.
- `--method`: Tokenization method (`sentencepiece`, `tiktoken`, `char`). Default is `sentencepiece`.
- `--vocab_size`: Vocabulary size for the SentencePiece model. Default is 500.

#### `prepare.py` Generated File Descriptions

Afterward it produces the train.bin and val.bin (and meta.pkl if not tiktoken)
* `train.bin` - training split containing 90% of the input file
* `val.bin` - validation split containing 10% of the input file
* `meta.pkl` - additonal file specifying tokenization needed for training and inference (Note: not produced/required for tiktoken)

These files above are then utilized to train the model via the `train.py` or
`run_experiments.py` wrapper scripts.

### (Optional) Pre-processing of input.txt

There are a number of methods to preprocess data before tokenization.

#### Phonemization

To experiment with utilization of a phonemized version of the dataset, first
convert the dataset into text format (e.g. `input.txt`), and run the
phonemization script on the text file:

```bash
bash txt_to_phonemes.sh input.txt phonemized_output.txt
```

The above by default utilizes all of the cores of one's machine to speed up the
process.

To specify the number of cores to utilize, use the following invocation, setting
the number of cores to limit the script to:

```bash
bash txt_to_phonemes.sh -n 8 input.txt phonemized_output.txt
```

The above command will limit the script to utilize only 8 cores at any one time.

## Relevant Tokenization Resources and References

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
