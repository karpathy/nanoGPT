from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load the Wikitext dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add a new pad token (if necessary)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the input text and create labels that are shifted versions of the input
    encodings = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
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
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
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
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("gpt2-finetuned-wikitext")
tokenizer.save_pretrained("gpt2-finetuned-wikitext")