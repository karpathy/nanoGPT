## OPUS-100 Dataset

This folder contains scripts compatible with the OPUS-100 dataset.

The OPUS-100 dataset is an English-centric multilingual corpus designed for the
task of translation, covering 100 languages. It is structured around English,
meaning that all language pairs include English either as the source or the
target language. The dataset comprises approximately 55M sentence pairs, with a
varied amount of training data per language pair, showcasing the diversity and
scale of the corpus.

## Languages

Out of the 99 language pairs available, 44 pairs have over 1M sentence pairs for
training, 73 have at least 100k, and 95 have at least 10k, demonstrating a
significant breadth of language coverage.

## Dataset Structure

A typical data instance in OPUS-100 looks like this:

```json
{
  "translation": {
    "ca": "El departament de bombers té el seu propi equip d'investigació.",
    "en": "Well, the fire department has its own investigative unit."
  }
}
```

## Citation

```
@inproceedings{zhang-etal-2020-improving,
    title = "Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation",
    author = "Zhang, Biao and Williams, Philip and Titov, Ivan and Sennrich, Rico",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.148",
    doi = "10.18653/v1/2020.acl-main.148",
    pages = "1628--1639",
}

@inproceedings{tiedemann-2012-parallel,
    title = "Parallel Data, Tools and Interfaces in {OPUS}",
    author = {Tiedemann, J{\"o}rg},
    booktitle = "Proceedings of the Eighth International Conference on Language Resources and Evaluation ({LREC}'12)",
    month = may,
    year = "2012",
    address = "Istanbul, Turkey",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf",
    pages = "2214--2218",
}
```

## Websites and More Information

More information can be found on the Huggingface Website
https://huggingface.co/datasets/Helsinki-NLP/opus-100

And the project webpage:
https://opus.nlpl.eu/OPUS-100
