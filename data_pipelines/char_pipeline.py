import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, data_file, block_size):
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+1+self.block_size].astype(np.int64))
        return x, y

class CharDataPipeline:
    def __init__(self, data_dir, batch_size, block_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size

        self.train_dataset = CharDataset(os.path.join(data_dir, 'train.bin'), self.block_size)
        self.val_dataset = CharDataset(os.path.join(data_dir, 'val.bin'), self.block_size)

    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def get_val_loader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def __iter__(self):
        # This is a simple infinite iterator for the training data
        while True:
            for x, y in self.get_train_loader():
                yield x, y
