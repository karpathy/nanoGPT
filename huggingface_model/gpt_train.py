from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import gpt_model

# Load the Wikitext dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Add a new pad token
gpt_model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt_model.pretrained_model.resize_token_embeddings(len(gpt_model.tokenizer))

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the input text and create labels that are shifted versions of the input
    encodings = gpt_model.tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
    input_ids = encodings['input_ids']
    labels = input_ids.copy()  # GPT-2 uses the same input for labels
    encodings['labels'] = labels
    return encodings

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the datasets for PyTorch
train_dataset = tokenized_datasets['train'].remove_columns(["text"]).with_format("torch")
eval_dataset = tokenized_datasets['validation'].remove_columns(["text"]).with_format("torch")

# Define training arguments
training_args = TrainingArguments(
    output_dir='results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="steps",
    logging_dir="./logs",
    logging_steps=500,
)

# Define the Trainer
trainer = Trainer(
    model=gpt_model.pretrained_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

gpt_model.pretrained_model.save_pretrained("gpt2-custom")
gpt_model.tokenizer.save_pretrained("gpt2-custom")