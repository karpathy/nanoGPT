# Zyda

This directory contains scripts compatible with the Zyda dataset.

## Description

Zyda is a 1.3T language modeling dataset created by collecting open and
high-quality datasets, combining them, and performing a uniform filtering and
deduplication step. Zyda performs extremely well in ablations and is at least
comparable to the best openly available datasets due to a meticulous
post-processing pipeline. Zyda can be used as a standalone dataset for language
model training up to the 1T scale or in combination with Fineweb or Dolma for
multi-trillion token training. For detailed information and citation, please
refer to the Zyphra team.

## Dataset Description

- **Curated by:** Zyphra
- **Language(s) (NLP):** Primarily English
- **License:** Open Data Commons License

## Dataset Structure

**Dataset fields:**

- `text`: Contains actual text for training
- `source`: Component the text is coming from
- `filtering_features`: Precomputed values of different features that were used for filtering (converted to JSON string)
- `source_other`: Metadata from the source dataset (converted to JSON string)

## Source Data

Zyda was drawn from seven component open datasets which are well-regarded in the community. These are:

- Pile Uncopyrighted: [https://huggingface.co/datasets/monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted)
- C4-en: [https://huggingface.co/datasets/allenai/c4](https://huggingface.co/datasets/allenai/c4)
- peS2o: [https://huggingface.co/datasets/allenai/peS2o](https://huggingface.co/datasets/allenai/peS2o)
- RefinedWeb: [https://huggingface.co/datasets/tiiuae/falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
- SlimPajama: [https://huggingface.co/datasets/cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
- arxiv_s2orc_parsed: [https://huggingface.co/datasets/ArtifactAI/arxiv_s2orc_parsed](https://huggingface.co/datasets/ArtifactAI/arxiv_s2orc_parsed)
- StarCoder: [https://huggingface.co/datasets/bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)

## Data Collection and Processing

Zyda was created using a two-stage post-processing pipeline consisting of filtering and deduplication.

- **Filtering Stage:** Utilized a set of hand-crafted and tuned filters derived from sources such as C4, RedPajama, and Gopher, in addition to Zyphra's own filters.
- **Deduplication Stage:** Used minhash approximate deduplication, deduplicating on 13-grams with a minhash signature size of 128, and filtered out documents above a Jaccard similarity of 0.4.

For full details on the data processing, see the Zyda technical report and dataset processing code.

## Links:

- Huggingface Page - https://huggingface.co/datasets/Zyphra/Zyda
- Arxiv Publication - https://arxiv.org/abs/2406.01981

## Citation

If you use the dataset to train a model, please cite:

```
@misc{tokpanov2024zyda,
      title={Zyda: A 1.3T Dataset for Open Language Modeling},
      author={Yury Tokpanov and Beren Millidge and Paolo Glorioso and Jonathan Pilault and Adam Ibrahim and James Whittington and Quentin Anthony},
      year={2024},
      eprint={2406.01981},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


