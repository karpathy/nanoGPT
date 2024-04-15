MMLU Benchmark Dataset
=======================

This folder contains scripts compatible with the MMLU (Massive Multitask
Language Understanding) benchmark dataset.

Introduction
------------
The MMLU benchmark is a collection of educational and assessment data designed
to evaluate language understanding capabilities across a diverse range of
subjects. This dataset includes questions from multiple academic and
professional fields, providing a broad spectrum for testing comprehension and
reasoning in language models.

Downloading the Dataset
-----------------------
To download the MMLU benchmark dataset, you can use the provided
`get_dataset.py` script. This script automates the process of scraping and
converting Parquet files from a specified URL to JSON and saves its contents to
a text file.

Here's how to use the script:

```bash
python3 get_dataset.py
```

Dataset Structure
-----------------

The dataset includes multiple-choice questions with the following features:

    question: The text of the question being asked.
    subject: The academic subject or field the question pertains to.
    choices: A list of possible answers to the question.
    answer: The index of the correct answer in the choices list.

The dataset covers numerous subjects including but not limited to abstract
algebra, anatomy, astronomy, business ethics, and virology. This diversity
supports comprehensive assessments of language understanding across different
knowledge domains.

Dataset Licensing Information
-----------------------------

MIT

Citation
--------

If you use the MMLU benchmark dataset in your research, please cite the following:

```bibtex
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

