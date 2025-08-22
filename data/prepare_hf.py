import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset for nanoGPT.")
    parser.add_argument("dataset", type=str, help="The name of the Hugging Face dataset to use (e.g., 'wikitext')")
    parser.add_argument("--config_name", type=str, default=None, help="The configuration name of the dataset (e.g., 'wikitext-2-raw-v1').")
    parser.add_argument("--val_size", type=float, default=0.0005, help="The proportion of the training set to use for validation if no validation split exists.")
    parser.add_argument("--text_column", type=str, default="text", help="The name of the column containing the text data.")
    args = parser.parse_args()

    # number of workers in .map() call
    num_proc = os.cpu_count() // 2 if os.cpu_count() else 1

    # number of workers in load_dataset() call
    num_proc_load_dataset = num_proc

    # output directory for the prepared data
    output_dir = os.path.join(os.path.dirname(__file__), args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # load the dataset
    print(f"Loading dataset '{args.dataset}' with config '{args.config_name}'...")
    try:
        dataset = load_dataset(args.dataset, args.config_name, num_proc=num_proc_load_dataset)
    except Exception as e:
        print(f"Failed to load dataset. If it's a private or gated dataset, make sure you are logged in with 'huggingface-cli login'.")
        print(e)
        return

    # handle datasets that don't have a validation split
    if "validation" not in dataset and "val" not in dataset:
        print(f"No validation split found. Creating one with {args.val_size * 100}% of the training data.")
        # take a small validation set from the training set
        dataset = dataset["train"].train_test_split(test_size=args.val_size, seed=2357, shuffle=True)
        dataset['val'] = dataset.pop('test') # rename the test split to val

    # get the tokenizer
    enc = tiktoken.get_encoding("gpt2")

    def process(example):
        ids = enc.encode_ordinary(example[args.text_column]) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    print("Tokenizing the dataset...")
    tokenized = dataset.map(
        process,
        remove_columns=[args.text_column],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        if split == "validation":
            split_filename = "val"
        elif split == "train":
            split_filename = "train"
        else:
            print(f"Skipping split '{split}'")
            continue

        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'{split_filename}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Finished writing {filename}")

if __name__ == '__main__':
    main()
