# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 16

# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
dataset = load_dataset("CarperAI/openai_summarize_tldr")



# class TLDRDataset(Dataset):
#     def __init__(self, split):
#         self.text = []
#         dataset = load_dataset(train_path, split=split)
#         for sample in dataset:
#             self.text.append(sample["prompt"] + sample["label"])
#         # if "valid" in train_path:
#         #     self.post_list = self.post_list[0:2000]
#         # self.tokenizer = tokenizer
#         # self.max_length = max_length
#         # self.input_ids = []
#         # self.attn_masks = []

#     def __len__(self):
#         return len(self.text)

#     def __getitem__(self, idx):
#         txt = self.text[idx]
#         # encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
#         # input_ids = torch.tensor(encodings_dict["input_ids"])
#         # attn_masks = torch.tensor(encodings_dict["attention_mask"])

#         return {
#             # "input_ids": input_ids,
#             # "attention_mask": attn_masks,
#             # "labels": input_ids,
#         }

# dataset = TLDRDataset(split="train")

train_text_list = []
for sample in dataset['train']:
    train_text_list.append(sample['prompt'] + sample['label'])
dataset['train'] = dataset['train'].add_column('text', train_text_list) # add the text column to the train dataset

dataset['val'] = dataset.pop('valid') # rename the valid dataset to val

val_text_list = []
for sample in dataset['val']:
    val_text_list.append(sample['prompt'] + sample['label'])
dataset['val'] = dataset['val'].add_column('text', val_text_list) # add the text column to the val dataset

dataset.pop('test') # remove the test dataset

split_dataset = dataset

# this results in:
# >>> split_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['prompt', 'label', 'text'],
#         num_rows: 116722
#     })
#     val: Dataset({
#         features: ['prompt', 'label', 'text'],
#         num_rows: 6447
#     })
# })

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text','prompt','label'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    print(f"writing {filename}...")
    idx = 0
    for example in tqdm(dset):
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()

# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')
