# Fineweb-Edu

This directory contains scripts compatible with the Fineweb-Edu dataset.

## Description

📚 FineWeb-Edu dataset consists of 1.3T tokens and 5.4T tokens
(FineWeb-Edu-score-2) of educational web pages filtered from 🍷 FineWeb dataset.
The work includes an educational quality classifier using annotations generated
by LLama3-70B-Instruct to retain the most educational web pages. FineWeb-Edu
outperforms FineWeb on popular benchmarks and shows the power of classifiers
trained on synthetic data.

## Dataset Curation

FineWeb-Edu uses Llama3-70B-Instruct to score 500k FineWeb samples for
educational quality on a scale from 0 to 5.

A threshold of 3 during the filtering process, retaining a high-level
educational pages. A fine-tuned Bert-like regression model was createdd using
these annotations, achieving an F1 score of 82%. Filtering out samples with
scores lower than 3 removed 92% of the dataset, leaving 1.3T educational tokens.

## License

The dataset is released under the Open Data Commons Attribution License (ODC-By)
v1.0 license. The use of this dataset is also subject to CommonCrawl's Terms of
Use.

## Citation

```
@software{lozhkov2024fineweb-edu,
  author = {Lozhkov, Anton and Ben Allal, Loubna and von Werra, Leandro and Wolf, Thomas},
  title = {FineWeb-Edu},
  month = May,
  year = 2024,
  url = {https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu}
}
```
