# BillSum Dataset Processor

This Python script is designed to automate the downloading, conversion, and processing of the "BillSum" dataset, which includes summarization of US Congressional and California state bills. The dataset features text, summaries, and titles of the bills, alongside metadata such as the number of characters in the text and summaries.

## Dataset Overview

The BillSum dataset provides a comprehensive collection of legislative documents for natural language processing tasks, specifically focused on summarization. It comprises US Congressional bills, US test bills, and California state test bills, making it a valuable resource for research and development in legal document analysis and summarization.

### Features

- **text**: Bill text.
- **summary**: Summary of the bills.
- **title**: Title of the bills. (Note: California bills do not have titles.)
- **text_len**: Number of characters in text.
- **sum_len**: Number of characters in summary.

### Data Splits

The dataset is divided into the following splits:

- **Train**: 18,949 US bills
- **ca_test**: 1,237 California state test bills
- **test**: 3,269 US test bills

For more details, visit the [BillSum dataset page on Hugging Face](https://huggingface.co/datasets/billsum).

## Getting Started

### 1. Install Requirements

- Python 3.x
- Pandas
- Requests

### 2. Retrieve Dataset

```bash
python3 get_dataset.py
```

This will get the dataset into a text file "input.txt", expanding the json files
in order of "title" then "article" then "summary".

### 3. Run Tokenization

```bash
python3 prepare.py --method tiktoken
```

After running the above command you should be ready to add this folder as a
dataset for training.


## Acknowledgments

See the following links for more information about BillSum:

* [Huggingface BillSum Page](https://huggingface.co/datasets/billsum)
* [Arxiv Paper: BillSum: A Corpus for Automatic Summarization of US Legislation (Kornilova and Eidelman, 2019)](https://arxiv.org/abs/1910.00523)

Billsum bibtex citation:
```
@misc{kornilova2019billsum,
    title={BillSum: A Corpus for Automatic Summarization of US Legislation},
    author={Anastassia Kornilova and Vlad Eidelman},
    year={2019},
    eprint={1910.00523},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
