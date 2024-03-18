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


## Example script usage

### Basic workflow

Get dataset:
```bash
python3 get_dataset.py -f en -t fr
```

Partition into smaller files if necessary:
```bash
python3 partition_file.py --input_file input.txt
```

Batch process into single set of train.bin and val.bin files:
```bash
python3 batch_prepare.py --input_dir interleaved_files --prepare_script prepare.py --tokenizer sentencepiece
```

### Phoneme workflow

Get dataset and split into to and from txt files:
```bash
python3 get_dataset.py -f en -t fr  --phonemize
```

Create phonemized texts:
```bash
bash txt_to_phonemes.sh -l en -o  from_input.txt en_pho.txt
bash txt_to_phonemes.sh -l fr -o  to_input.txt fr_pho.txt
```

Train sentencepiece, skipping tokenization:
```bash
cat en_pho.txt fr_pho.txt > enfr_pho.txt
python3 prepare.py -t enfr_pho.txt --method sentencepiece --vocab_size 2048 --skip_sp_tokenization
```

Interleave outputs:
```bash
python3 interleave_files.py -f1 en_pho.txt -f2 fr_pho.txt -m 50 -o test --forbidden_strings "(en)" "(fr)"
```

Batch process into single set of train.bin and val.bin files:
```bash
python3 batch_prepare.py --input_dir interleaved_files --prepare_script prepare.py --tokenizer sentencepiece
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
