# Korean-Parallel-Corpora

This folder contains scripts and resources for working with the
korean-parallel-corpora dataset.

The Moo/korean-parallel-corpora is designed for tasks related to language
translation, specifically translating between Korean and English.

## Dataset Description:
- **URL:** [Moo/korean-parallel-corpora on Huggingface](https://huggingface.co/datasets/Moo/korean-parallel-corpora/discussions/1)
- **Tasks:** Translation
- **Languages:** Korean, English
- **License:** [cc-by-sa-3.0](https://creativecommons.org/licenses/by-sa/3.0/)

## Script Overview:
Scripts specific to this dataset are described below.

### Obtaining dataset with `get_dataset.py`
To obtain the dataset locally:

```bash
python3 get_dataset.py
```

This will create the following files:
- `data.json` - json containing Korean, English, and jamon language entries
- `input.txt` - text format with prefixed entries according to args sent to the `get_dataset.py` file.

### Testing Korean <-> Jamon Conversion

This script is an example of how to convert to jamon and back:
```bash
python3 korean_jamo_conversion_test.py
```
This will allow the model to be optionally trained on jamon but still recover
the original Korean text.

## License:

The dataset is listed at Huggingface under the Creative Commons Attribution-ShareAlike 3.0 License.

