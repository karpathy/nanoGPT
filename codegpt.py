import datasets
from datasets import load_dataset
from transformers import GPT2Tokenizer

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("Fsoft-AIC/the-vault-function", split_set=["train/small"], languages=['python'])

    # Display a sample
    print(dataset['train_small'][0])
    # print(type(dataset['train_small']))
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def preprocess_function(examples):
        try:
            # Assume examples is a dictionary with lists of values
            inputs = examples['code']
            targets = examples['docstring']
        except TypeError:
            inputs = [ex['code'] for ex in examples]
            targets = [ex['docstring'] for ex in examples]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
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