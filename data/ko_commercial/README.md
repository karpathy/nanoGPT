# KoCommercial-Dataset

## Overview

The KoCommercial-Dataset is a comprehensive collection of approximately 1.44
million data points, released under the MIT License. This dataset amalgamates
various Korean language resources aimed at fostering advancements in natural
language processing and machine learning fields.

## License

This dataset is listed as MIT License.

## Dataset Composition

The KoCommercial-Dataset encompasses a wide range of sources, including but not
limited to:

- `kyujinpy/KOpen-platypus` (*Excluding non-commercial datasets)
- `beomi/KoAlpaca-v1.1a`
- `HumanF-MarkrAI/WIKI_QA_Near_dedup`
- `KorQuadv1.0`
- `AIHUB`
    - General knowledge sentence generation data
    - Book material summaries
    - Research paper summaries
    - Document summary texts

### Self-Supervised Method (Processed AIHUB Dataset)

The dataset employs self-supervised methods to enhance its utility, particularly through:
- Summary & Instruction-Answer formats
- Sentence order inference
- Original sentence inference
- Last sentence prediction
- Multi-question answering
- Mask prediction

## Download and Usage

The scripts contained will download the dataset and parse into an input.txt
file.

To download:
```py
python3 get_dataset.py
```

Please follow the specific instructions and guidelines provided by each dataset
repository for optimal use.

## References

1. The CoT Collection: Improving Zero-shot and Few-shot Learning of Language Models via Chain-of-Thought Fine-Tuning (Kim et al., 2023)
2. Adapting Large Language Models via Reading Comprehension (Cheng et al., 2023)
3. Deduplicating Training Data Makes Language Models Better (Lee et al., 2021)

## Link to KoCommerical Huggingface Page

- https://huggingface.co/datasets/MarkrAI/KoCommercial-Dataset#dataset-kocommercial-dataset

## Acknowledgements

This project has been supported by the Artificial intelligence industrial
convergence cluster development project funded by the Ministry of Science and
ICT(MSIT, Korea) & Gwangju Metropolitan City. We extend our gratitude to the
open-source developers and researchers contributing to this field, and a special
thanks to Beomi and maywell for their significant contributions to the Korean
LLM open ecosystem.

