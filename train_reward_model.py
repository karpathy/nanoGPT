
from trainers.reward_trainer import RewardModelTrainer
import yaml

from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
import tiktoken
import torch


# with inspiration from CarperAI's trlx library

with open('config/config_reward.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
    # nested dictionary structure
    config = {}               
    for k, v in conf.items():
        for k2, v2 in v.items():
            config[k2] = v2
print(config)

def create_comparison_dataset(path="CarperAI/openai_summarize_comparisons", split="train"):
    dataset = load_dataset(path, split=split)
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, max_length):
        # self.chosen_input_ids = []
        # self.chosen_attn_masks = []
        # self.rejected_input_ids = []
        # self.rejected_attn_masks = []
        self.chosens = []
        self.rejecteds = []
        self.enc = tiktoken.get_encoding("gpt2")
        for pair in tqdm(pairs):
            chosen_enc = self.enc.encode("<|startoftext|>" + pair['chosen'] + "<|endoftext|>", allowed_special="all")[-max_length:]
            rejected_enc = self.enc.encode("<|startoftext|>" + pair['rejected'] + "<|endoftext|>", allowed_special="all")[-max_length:]
            self.chosens.append(chosen_enc)
            self.rejecteds.append(rejected_enc)  

    def __len__(self):
        return len(self.chosens)
    
    def __getitem__(self, idx):
        return (
            self.chosens[idx],
            self.rejecteds[idx],
        )

class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["chosen_ids"] = torch.tensor([f[0] for f in data])
        batch["rejected_ids"] = torch.tensor([f[1] for f in data])
        return batch

def collate_fn(data):
    batch = {}
    batch["chosen_ids"] = torch.tensor([f[0] for f in data])
    batch["rejected_ids"] = torch.tensor([f[1] for f in data])
    return batch   

data_path = "CarperAI/openai_summarize_comparisons"
train_pairs = create_comparison_dataset(data_path, "train")
val_pairs = create_comparison_dataset(data_path, "test")


# Make pairwise datasets for training
print("Creating pairwise datasets")
train_dataset = PairwiseDataset(train_pairs, max_length=config['block_size'])
val_dataset = PairwiseDataset(val_pairs, max_length=config['block_size'])

trainer = RewardModelTrainer(config, train_dataset, val_dataset, collate_fn=collate_fn)

trainer.train()

