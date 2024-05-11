# CNN/DailyMail Dataset Processor

This Python script is designed to automate the downloading, conversion, and
processing of the "CNN/DailyMail" dataset, which is an extensive collection of
news articles from CNN and the Daily Mail for summarization tasks. The dataset
supports both extractive and abstractive summarization methods, containing over
300k unique news articles.

## Dataset Overview

The CNN/DailyMail dataset is an English-language dataset focused on
summarization. It contains news articles and highlights, enabling the
development of models for abstractive and extractive summarization. The
dataset's performance is often measured by the ROUGE score, comparing the output
summary to the highlights written by the original authors.

### Features

- **article**: The body of the news article.
- **highlights**: The highlight of the article as written by the article author.
- **id**: A string containing the heximal formatted SHA1 hash of the URL where the story was retrieved from.

## Getting Started

### Download and Process Dataset

Run the provided script to download the Parquet files, convert them to a
readable format, and save the processed data into `input.txt` by default.

```bash
python3 get_dataset.py
```

### Tokenize data

```bash
python3 prepare.py --method tiktoken
```
The above will process the `input.txt` into files required to begin training on
this data.

Only the article and highlights are included and prefixed with:

```
text: article content
summary: 'highlights' content
```

## Attribution

The CNN / Daily Mail dataset was released under the Apache-2.0 License.

For more details, visit the [CNN/DailyMail dataset page on Hugging Face](https://huggingface.co/datasets/cnn_dailymail).

Citation information follows:
```
@inproceedings{see-etal-2017-get,
    title = "Get To The Point: Summarization with Pointer-Generator Networks",
    author = "See, Abigail  and
      Liu, Peter J.  and
      Manning, Christopher D.",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P17-1099",
    doi = "10.18653/v1/P17-1099",
    pages = "1073--1083",
    abstract = "Neural sequence-to-sequence models have provided a viable new approach for abstractive text summarization (meaning they are not restricted to simply selecting and rearranging passages from the original text). However, these models have two shortcomings: they are liable to reproduce factual details inaccurately, and they tend to repeat themselves. In this work we propose a novel architecture that augments the standard sequence-to-sequence attentional model in two orthogonal ways. First, we use a hybrid pointer-generator network that can copy words from the source text via pointing, which aids accurate reproduction of information, while retaining the ability to produce novel words through the generator. Second, we use coverage to keep track of what has been summarized, which discourages repetition. We apply our model to the CNN / Daily Mail summarization task, outperforming the current abstractive state-of-the-art by at least 2 ROUGE points.",
}

@inproceedings{DBLP:conf/nips/HermannKGEKSB15,
  author={Karl Moritz Hermann and Tomás Kociský and Edward Grefenstette and Lasse Espeholt and Will Kay and Mustafa Suleyman and Phil Blunsom},
  title={Teaching Machines to Read and Comprehend},
  year={2015},
  cdate={1420070400000},
  pages={1693-1701},
  url={http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend},
  booktitle={NIPS},
  crossref={conf/nips/2015}
}
```

