# Wikipedia Dataset Processing

## Overview
This project focuses on processing subsets of the Wikipedia dataset for multiple languages, including English (en), German (de), North Frisian (frr), French (fr), Italian (it), and Simple English (simple). The processing involves downloading dataset files, converting them from Parquet to JSON, and emitting the text content of the articles into language-specific text files.

## Dataset Structure
Each data instance in the processed files contains the following fields:

- `id`: ID of the article
- `url`: URL of the article
- `title`: Title of the article
- `text`: Text content of the article

## Data Splits and Size
The dataset is split into training data for several languages with varying sizes:

- **German (de)**: 2,665,357 articles
- **English (en)**: 6,458,670 articles
- **French (fr)**: 2,402,095 articles
- **North Frisian (frr)**: 15,199 articles
- **Italian (it)**: 1,743,035 articles
- **Simple English (simple)**: 205,328 articles

The disk usage for each language, including downloaded dataset files and generated datasets, is as follows:

- **German (de)**: Total of 14.25 GB
- **English (en)**: Total of 31.96 GB
- **French (fr)**: Total of 11.60 GB
- **North Frisian (frr)**: Total of 13.66 MB
- **Italian (it)**: Total of 7.25 GB
- **Simple English (simple)**: Total of 368.96 MB

## Usage
To process the datasets, run the provided Python script. The script will automatically handle the downloading, conversion, and text emission for each specified language. The output will be language-specific text files named as follows:

- `en.input.txt`
- `de.input.txt`
- `fr.input.txt`
- `frr.input.txt`
- `it.input.txt`
- `simple.input.txt`

Each file contains the text content of the Wikipedia articles for the respective language.

## Licensing Information

From the [huggingface page](https://huggingface.co/datasets/wikipedia):

```
Most of Wikipedia's text and many of its images are co-licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License (CC BY-SA) and the GNU Free Documentation License (GFDL) (unversioned, with no invariant sections, front-cover texts, or back-cover texts).

Some text has been imported only under CC BY-SA and CC BY-SA-compatible license and cannot be reused under GFDL; such text will be identified on the page footer, in the page history, or on the discussion page of the article that utilizes the text.
```

## Citation Information

If you use this dataset for your research, please cite the following:

```
@ONLINE{wikidump,
    author = "Wikimedia Foundation",
    title  = "Wikimedia Downloads",
    url    = "https://dumps.wikimedia.org"
}
```
