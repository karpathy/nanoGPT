import os
import argparse
import numpy as np
import pickle
import sentencepiece as spm
import tiktoken


def train_sentencepiece_model(input_files, model_prefix, vocab_size):
    """Train a SentencePiece model."""

    # Other options (https://github.com/google/sentencepiece/blob/master/doc/options.md)
    # self_test_sample_size=1,
    # input_format="text",
    # shuffle_input_sentence = false
    # split_digits=False, # this often helps with arithmetic
    # allow_whitespace_only_pieces=True,
    # normalization_rule_name="nmt_nfkc_cf" lower cases as well
    num_threads = os.cpu_count()
    input_arg = ",".join(input_files) if isinstance(input_files, list) else input_files
    spm.SentencePieceTrainer.train(
        num_threads=num_threads,
        input=input_arg,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
    )
    print("SentencePiece model training complete.")


def tokenize_sentencepiece(sp_model, data):
    """Tokenize data using the SentencePiece model."""
    return sp_model.encode_as_ids(data)


def tokenize_tiktoken(enc, data):
    """Tokenize data using TikToken."""
    return enc.encode_ordinary(data)


def encode_char_level(data, chars):
    """Encode data at character level."""
    stoi = {ch: i for i, ch in enumerate(chars)}
    return [stoi[ch] for ch in data], stoi, {i: ch for i, ch in enumerate(chars)}


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize text data using different methods."
    )

    # Default option is a single input file
    parser.add_argument('-i', "--input_file", type=str, help="Path to the input text file")

    parser.add_argument(
        "--method",
        type=str,
        choices=["sentencepiece", "tiktoken", "char"],
        default="tiktoken",
        help="Tokenization method",
    )

    # Sentence Piece only argument
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=500,
        help="Vocabulary size for SentencePiece model",
    )

    # Customize output names for bins
    parser.add_argument(
        "--train_output",
        type=str,
        default="train.bin",
        help="Output file for tokenized training data",
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="val.bin",
        help="Output file for tokenized validation data",
    )

    # Options for using separate training and validation input files
    parser.add_argument(
        "-s",
        "--use_separate_files",
        action="store_true",
        help="Use separate files for training and validation input",
    )
    parser.add_argument(
        "-t", "--train_input", type=str, help="Path to the training input text file"
    )
    parser.add_argument(
        "-v", "--val_input", type=str, help="Path to the validation input text file"
    )

    args = parser.parse_args()

    if args.use_separate_files:
        if not args.train_input or not args.val_input:
            raise ValueError(
                "Both --train_input and --val_input must be provided when using --use_separate_files."
            )
        input_files = [args.train_input, args.val_input]
    else:
        if not args.train_input:
            raise ValueError(
                "You must provide --train_input when not using --use_separate_files."
            )
        input_files = args.train_input

    # Read data
    if args.use_separate_files:
        with open(args.train_input, "r") as f:
            train_data = f.read()
        with open(args.val_input, "r") as f:
            val_data = f.read()
    else:
        with open(args.train_input, "r") as f:
            data = f.read()
        n = len(data)
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9) :]

    if args.method == "sentencepiece":
        # Train and use SentencePiece
        spm_model_prefix = os.path.splitext(args.input_file)[0] + "_spm_model"
        train_sentencepiece_model(input_files, spm_model_prefix, args.vocab_size)
        sp = spm.SentencePieceProcessor()
        sp.load(f"{spm_model_prefix}.model")
        train_ids = tokenize_sentencepiece(sp, train_data)
        val_ids = tokenize_sentencepiece(sp, val_data)

        # Create stoi (string-to-index) and itos (index-to-string) mappings
        stoi = {sp.id_to_piece(id): id for id in range(sp.GetPieceSize())}
        itos = {id: sp.id_to_piece(id) for id in range(sp.GetPieceSize())}

        # Manually add newline character to vocab
        if "\n" not in stoi:
            stoi["\n"] = sp.PieceToId("\n")

        # Save metadata including stoi and itos in a pickle file
        meta = {"vocab_size": sp.GetPieceSize(), "stoi": stoi, "itos": itos}
        with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    elif args.method == "tiktoken":
        # Use TikToken
        enc = tiktoken.get_encoding("gpt2")
        train_ids = tokenize_tiktoken(enc, train_data)
        val_ids = tokenize_tiktoken(enc, val_data)

    elif args.method == "char":
        # Print the total length of the dataset in characters
        print(f"Length of dataset in characters: {len(data):,}")
        # Character-level tokenization
        chars = sorted(
            list(set(train_data + val_data))
        )  # Get unique characters in train data
        vocab_size = len(chars)
        print("All unique characters:", "".join(chars))
        print(f"Vocab size: {vocab_size}")

        train_ids, stoi, itos = encode_char_level(train_data, chars)
        val_ids, _, _ = encode_char_level(val_data, chars)

        # Save the meta information
        meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
        with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    # Print token counts and export to bin files
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    np.array(train_ids, dtype=np.uint16).tofile(args.train_output)
    np.array(val_ids, dtype=np.uint16).tofile(args.val_output)


if __name__ == "__main__":
    main()
