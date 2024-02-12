from datasets import load_dataset, concatenate_datasets
import json

# Load the XSum dataset
raw_datasets = load_dataset("EdinburghNLP/xsum")

# Rename and remove columns as necessary
processed_datasets = {}
for split, dataset in raw_datasets.items():
    # Rename 'document' column to 'text' and remove 'id' column
    dataset = dataset.rename_column("document", "text")
    dataset = dataset.remove_columns(["id"])

    # Save processed dataset to JSON file
    json_path = f"xsum-{split}.json"
    dataset.to_json(json_path)

    # Store processed dataset for later use
    processed_datasets[split] = dataset

# Combine datasets if needed
combined_dataset = concatenate_datasets([dataset for dataset in processed_datasets.values()])
combined_dataset.to_json("xsum-combined.json")

print("Dataset processing and saving to JSON completed.")
def save_dataset_to_text(dataset, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in dataset:
            text = item['text']
            summary = item['summary']
            file.write(f"text: {text}\nsummary: {summary}\n\n")

# Create a combined dataset for xsum
save_dataset_to_text(combined_dataset, "xsum-combined.txt")

print("Dataset conversion to text completed.")

