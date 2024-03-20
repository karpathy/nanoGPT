# CC-100 Dataset Utilities

This repository contains scripts designed to facilitate the use of the CC-100
dataset. The CC-100 corpus is a collection of monolingual datasets for over 100
languages, derived from web crawl data. It aims to recreate the dataset used for
training the XLM-R model and includes data for romanized languages (indicated by
*_rom). The data was compiled from January to December 2018 Commoncrawl
snapshots, using URLs and paragraph indices provided by the CC-Net repository.

## Dataset Description

The CC-100 corpus is an extensive multilingual resource constructed using the
open-source CC-Net repository. It features monolingual data for more than 100
languages, including romanized versions for certain languages. Documents in the
dataset are separated by double-newlines, with paragraphs within the same
document separated by a single newline. This structure aids in the easy
processing and utilization of the corpus for various natural language processing
(NLP) tasks.

## Usage

### Downloading and Extracting Data

The provided scripts enable users to download and extract specific language
datasets from the CC-100 corpus. Users can specify a particular language using
its language code or opt to download and extract all available languages.

#### Download a Specific Language Dataset

Run the script with the `-c` or `--code` option followed by the language code:

```bash
python3 get_dataset.py -c [language_code]
```

Replace [language_code] with the desired language's code (e.g., en for English, zh_Hans for Simplified Chinese).
Download All Available Languages

To download and extract datasets for all available languages, use the all option:


```bash
python3 get_dataset.py -c all
```

## Script Requirements

The scripts require Python 3.x and the following libraries:

- requests
- tqdm
- lzma

Ensure you have these dependencies installed by running:

```bash
python3 -m pip install requests tqdm
```
`lzma` is included in the standard library for Python 3.x.

## References

If you use the resources provided in this corpus, please cite the following works:

```
Unsupervised Cross-lingual Representation Learning at Scale, Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), p. 8440-8451, July 2020. PDF | BibTeX
CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data, Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, Edouard Grave, Proceedings of the 12th Language Resources and Evaluation Conference (LREC), p. 4003-4012, May 2020. PDF | BibTeX
```

For more details about the CC-100 dataset, visit https://data.statmt.org/cc-100/.
