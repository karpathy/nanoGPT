# Modular Arithmetic

Modular arithmetic example generatoion, printed in different bases for experimentation.

The purpose of this is to explore, demonstate and provide replicable recipes for
various training and validation curve effects within a simple space.

This also serves as an easy target to practice training reinforcement learning
scripts that are aware of the 'grokking' phenomenon, and test incremental or
progressive learning (advancing only upon mastery).

## Files

- `tokens.txt` - Contains a set of characters to use for encoding
- `prepare.py` - Script to encode the raw examples into ids and save in binary format
- `print_bases_mod_x.py` - Prints modulo arithmetic examples in different bases
- `create_examples.sh` - Runs `print_bases_mod_x.py` to generate the raw examples
- `data/` - Directory containing the raw arithmetic example files

## Creating Examples and Selecting Tokenisation Space

1. Run `create_examples.sh` to generate the raw examples in `data/`

2. Include any tokens you want to include in `tokens.txt`, for example:

```
 0123456789abcef

```

Above would have space, and newlines as characters in addition to base 16 chars.

3. Process the examples into a training set:
```bash
python3 prepare.py --input_file data/base_16.txt --token_file tokens.txt
```

This will encode the raw text into ids and save train/val splits along with vocab files in the current directory.

These files can then be used to train modular arithmetic models.

## Creating Tokenisation Straight From Text Files

If the `token_file` flag is omitted from the invocation, then the tokenization
is drawn from the `input_file`.

For example:
```bash
python3 prepare.py --input_file data/base_2.txt
```

Would only contain spaces, newlines, and characters 1 and 0 for binary
arithmetic.


## Details

- `print_bases_mod_x.py` generates examples like:

```
1 1 2
```

Which represents `1 % 16 + 1 % 16 = 2 (mod 16)` in base 16.

- Bases 1, 2, 4, 8, and 16 are provided.
- `prepare.py` encodes the raw text into ids using the char vocab in `tokens.txt`.
- The data is split 90/10 into train/val.
- Vocab and encoding mappings are saved to disk for future decoding.

## Options

* `--base` - e.g. `--base 16` - base to use {1,2,4,8,16}
* `--modulo` - e.g. `--modulo 101` - modulo to utilize and total number of numbers
* `--seed` - e.g. `--seed 42` - random seed to utilize, this way the ordering of examples in the different generated files can actually be the same
* `--no_separator` - if selected, then do not separate numbers with spaces
* `--little_endian` - if selected, reverses order of numbers (testing if this makes it easier for the transformer to predict the algorithm, being able to do it more stepwise.)

