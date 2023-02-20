
from trainer_reward import RewardModelTrainer
import yaml

from datasets import load_dataset
from torch.utils.data import Dataset
# assert enc.decode(enc.encode("hello world")) == "hello world"
from tqdm import tqdm
import tiktoken
import torch

with open('config_reward.yaml') as f:
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

            # chosen_encodings_dict = self.enc(
            #     "<|startoftext|>" + chosen + "<|endoftext|>",
            #     truncation=True,
            #     max_length=max_length,
            #     padding="max_length",
            #     return_tensors="pt",
            # )
            # rejected_encodings_dict = self.enc(
            #     "<|startoftext|>" + rejected + "<|endoftext|>",
            #     truncation=True,
            #     max_length=max_length,
            #     padding="max_length",
            #     return_tensors="pt",
            # )
            # self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            # self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            # self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            # self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        # return len(self.chosen_input_ids)
        return len(self.chosens)
    
    def __getitem__(self, idx):
        return (
            self.chosens[idx],
            self.rejecteds[idx],
        )

    # def __getitem__(self, idx):
    #     return (
    #         self.chosen_input_ids[idx],
    #         self.chosen_attn_masks[idx],
    #         self.rejected_input_ids[idx],
    #         self.rejected_attn_masks[idx],
    #     )

class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        # batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        # batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        # batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
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

