import re
import random
from typing import Dict, Tuple
import torch
import os

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import numpy as np




# Custom tokenizer with a placeholder list of tokens
class SATTokenizer(PreTrainedTokenizer):
    def __init__(self, max_id = 30, **kwargs):
        vocab_list = [str(i) for i in range(max_id + 1)] + [str(-i) for i in range(1, max_id + 1)] + ["[SEP]", "SAT", "UNSAT", "[EOS]", "Decide", "[UP]", "D", "[BT]", "[UNK]"]
        self.unk_token = "[UNK]"
        self.vocab = {v: k for k, v in enumerate(vocab_list)}
        self.ids_to_tokens = {k: v for v, k in self.vocab.items()}
        super().__init__(**kwargs)
        self.add_special_tokens({'pad_token': '[EOS]', 'eos_token': '[EOS]'})

    @classmethod
    def from_pretrained(cls, dir):
        filename = os.path.join(dir, "vocab.txt")
        vocab_list = []
        with open(filename, "r", encoding="utf-8") as reader:
            for line in reader:
                vocab_list.append(line.strip())
        return cls(vocab_list)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token)

    def _convert_id_to_token(self, id):
        return self.ids_to_tokens.get(id)

    def tokenize(self, text, **kwargs):
        return text.split()

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> Tuple[str]:
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token in self.vocab.keys():
                writer.write(token + "\n")
        return (vocab_file,)

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return len(self.vocab)


class SATDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size, max_id, split='train', val_ratio=0.01,
                 remove_trace=False, shift_within_block=False, permute_constants=True, mask_formula=True, old_tokenizer=False):
        self.tokenizer = tokenizer
        self.max_id = max_id
        self.block_size = block_size
        self.shift_within_block = shift_within_block
        self.permute_constants = permute_constants
        self.mask_formula = mask_formula
        self.old_tokenizer = old_tokenizer

        self.examples = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip().replace("-", "- ")
                if len(line) > 0:
                    if remove_trace:
                        # Remove the trace if requested
                        line = line[:line.find("[SEP]") + len("[SEP]")] + " " + line[line.find("UNSAT") if "UNSAT" in line else line.find("SAT"):]
                    self.examples.append(line)

        # Determine split indices
        dataset_size = len(self.examples)
        val_size = int(dataset_size * val_ratio)
        if val_size < 1 and dataset_size > 0:
            val_size = 1  # ensure at least one sample for validation if dataset is small

        if split == 'train':
            self.examples = self.examples[:-val_size] if val_size < dataset_size else []
        else:
            self.examples = self.examples[-val_size:]

    def __len__(self):
        return len(self.examples)

    def left_pad(self, tensor, pad_size, pad_with, block_size):
        p = torch.tensor([pad_with] * pad_size)
        return torch.cat((p, tensor))[0:block_size]

    def multiple_replace(self, string, rep_dict):
        pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]), flags=re.DOTALL)
        return pattern.sub(lambda x: rep_dict[x.group(0)], string)

    def constant_permuter(self, line):
        constants = list(range(1, self.max_id + 1))
        permutation = random.sample(constants, len(constants))
        permutation = [0] + permutation  # 0 is a special token

        return " ".join([str(permutation[int(tok)]) if tok.isdigit() else tok for tok in line.split()])

    def __getitem__(self, i):
        line = self.examples[i]

        if self.permute_constants:
            line = self.constant_permuter(line)

        if not self.old_tokenizer:
            # Remove spaces after negation symbol to match expected tokenizer format
            line = line.replace("- ", "-")

        tokens = self.tokenizer(line, truncation=True, padding="max_length", max_length=self.block_size, return_tensors="pt")
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()

        labels = input_ids.clone()
        labels[labels == pad_token_id] = -100

        if self.mask_formula:
            sep_token_id = self.tokenizer.encode("[SEP]")[0]
            sep_pos = torch.nonzero(input_ids == sep_token_id)
            if sep_pos.numel() > 0:
                labels[:sep_pos[0][0]] = -100

        if self.shift_within_block:
            input_len = torch.sum(attention_mask)
            size_left_pad = torch.randint(high=(self.block_size - input_len) if self.block_size > input_len else 0, size=(1,)).item()
            input_ids = self.left_pad(input_ids, size_left_pad, pad_token_id, self.block_size)
            labels = self.left_pad(labels, size_left_pad, -100, self.block_size)
            attention_mask = self.left_pad(attention_mask, size_left_pad, 0, self.block_size)

        return {"input_ids": input_ids.long(), "labels": labels.long(), "attention_mask": attention_mask}

class SATDatasetForRL(SATDataset):
    def __getitem__(self, i):
        line = self.examples[i]

        block_size = self.block_size

        if self.permute_constants:
            # If this option is invoked, come up with a random permutation of
            # the constants in the CNF formula and replace them in the line
            # for this SAT problem only
            line = self.constant_permuter(line)

        if not self.old_tokenizer:
            line = line.replace("- ", "-")

        split = line.find("[SEP]") + len("[SEP]")
        query = line[:split]
        response = line[(split + 1):]

        return {"query": query, "expected_response": response}


if __name__ == "__main__":
    # Test the dataset
    max_id = 30
    custom_tokens = [str(i) for i in range(max_id + 1)] + ["-", "[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]"]

    # custom_tokens = [str(i) for i in range(max_id + 1)] + [str(-i) for i in range(1, max_id + 1)] + ["[SEP]", "SAT", "UNSAT", "[EOS]", "[UNK]", "(", ")"]

    tokenizer = SATTokenizer(custom_tokens)
    tokenizer.add_special_tokens({'pad_token': '[EOS]'})
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})

    print(tokenizer.eos_token_id)

    block_size = 800

    DATASET_PATH = "./datasets/SAT_6_10_Marginal_Old/train.txt"
    dataset = SATDatasetForRL(
        DATASET_PATH,
        tokenizer,
        block_size,
        max_id,
        old_tokenizer=True,
        permute_constants=True
    )

    # i = torch.randint(0, len(dataset), (1,)).item()
    i = 0
    test_item = dataset.__getitem__(i)

    print(test_item)
    print(tokenizer.encode(test_item["query"], return_tensors="pt"))

    print(enumerate)