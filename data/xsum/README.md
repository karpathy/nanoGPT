# XSum Dataset Processor

This Python script is designed to automate the downloading, conversion, and processing of the "xsum" dataset, which includes 226,711 news articles accompanied with a one-sentence summary.

## Dataset Overview

The Extreme Summarization (XSum) dataset contains 226,711 BBC news articles (2010-2017) accompanied with a one-sentence summary. The articles cover a wide variety of domains (e.g., News, Politics, Sports, Weather, Business, Technology, Science, Health, Family, Education, Entertainment and Arts). For each article, a sentence is provided that is intented to explain what the article is about.

## Getting Started

### 1. Install Requirements

- Python 3.x
- NumPy
- huggingface
- json

### 2. Retrieve Dataset

```bash
python3 get_dataset.py
```

This will get the dataset into a text file "input.txt", dividing each article and summary into "text" and "summary" denoted sections.

### 3. Run Tokenization

```bash
python3 prepare.py --method tiktoken
```

After running the above command you should be ready to add this folder as a
dataset for training.


## Acknowledgments

See the following links for more information about XSum:

* [Huggingface XSum Page](https://huggingface.co/datasets/EdinburghNLP/xsum)
* [Arxiv Paper: Don’t Give Me the Details, Just the Summary!
Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://arxiv.org/pdf/1808.08745v1)

xsum bibtex citation:
```
@misc{Narayan_Cohen_Lapata_2018,
  title={Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization},
  author={Shashi Narayan, Shay B. Cohen, Mirella Lapata},
  year={2018},
  eprint={1808.08745},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
```
