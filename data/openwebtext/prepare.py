import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Number of workers in .map() call
num_proc = 32

# Number of workers in load_dataset() call
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("openwebtext",trust_remote_code=True, num_proc=num_proc_load_dataset)

    # Create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    # Tokenize the dataset
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Calculate the number of tokens in the validation set
    val_token_count = np.sum(tokenized['val']['len'], dtype=np.uint64)

    # Determine the number of tokens needed for the training set
    target_train_tokens = 4_000_000_000

    # Calculate cumulative token lengths for the training set
    train_lengths = np.cumsum(tokenized['train']['len'], dtype=np.uint64)
    
    # Find the index where the cumulative length is just over the target
    cutoff_index = np.searchsorted(train_lengths, target_train_tokens, side='right')

    # Create a new split with the reduced training set
    reduced_train_dataset = tokenized['train'].select(range(cutoff_index))

    # Concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in {'train': reduced_train_dataset, 'val': tokenized['val']}.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # To read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')