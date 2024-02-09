from datasets import Dataset, Features, Value, ClassLabel, concatenate_datasets
from datasets import load_dataset
import pandas as pd
import json

# Trying to use hugging face to download xsum
raw_datasets = load_dataset("EdinburghNLP/xsum")
for split, dataset in raw_datasets.items():
    dataset = dataset.rename_column("document", "text")
    dataset = dataset.remove_columns(["id"])
    dataset.to_json(f"xsum-{split}.json")
    raw_datasets[split] = dataset

print(raw_datasets)
combined_dataset = concatenate_datasets([dataset for dataset in raw_datasets.values()])
print(combined_dataset)
combined_dataset.to_json("xsum.json")

txt_types = ["-train", "-validation", "-test", ""]
for suffix in txt_types:
    fp = f"xsum{suffix}.json"
    output_fn = f"xsum{suffix}.txt"
    with open(fp) as inputf:
        with open(output_fn, 'w') as outputf:
            for line in inputf:
                line = line.rstrip()
                line = line.replace("{", "")
                line = line.replace("}", "")
                line = line.replace("\"", "")
                line = line.replace("\\n", " ")
                line = line.replace(",summary:", " summary:")

                json.dump(line, outputf)
                outputf.write('\n')
print('Json files successfully converted to txt')