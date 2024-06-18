import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import datasets
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Load the dataset
dataset = load_dataset("Fsoft-AIC/the-vault-function", split_set=["train/small"], languages=['python'])

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    try:
        # Assume examples is a dictionary with lists of values
        inputs = examples['code']
        targets = examples['docstring']
    except TypeError:
        inputs = [ex['code'] for ex in examples]
        targets = [ex['docstring'] for ex in examples]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
    model_inputs['labels'] = labels
    return model_inputs

tokenized_dataset = dataset['train_small'].map(preprocess_function, batched=True)

class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset):
        self.input_ids = tokenized_dataset["input_ids"]
        self.labels = tokenized_dataset["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "labels": torch.tensor(self.labels[idx])
        }
        return item

train_dataset = CodeDataset(tokenized_dataset)

# Hyperparameters
vocab_size = tokenizer.vocab_size
embed_size = 256
num_heads = 8
num_layers = 4
block_size = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CodeGPT(vocab_size, embed_size, num_heads, num_layers, block_size).to(device)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 3
model.train()

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids)
        loss = F.cross_entropy(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
    loss = loss.to("cpu")
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "gpt_model.pth")