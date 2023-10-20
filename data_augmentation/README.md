# Dataset Synthesis and Augmentation

Better datasets have been shown to reduce model sizes by 100x (see
[InstructGPT](https://arxiv.org/abs/2203.02155) paper).

Both [augmented](https://arxiv.org/abs/2309.09530) and fully synthetic [fully
synthetic](https://arxiv.org/abs/2305.07759) datasets have been demonstrated to
improve model quality by orders of magnitude, and provided here are means to
create and augment datasets using fully open source tools and models.

## TOC

* [Download TinyStories Dataset](#download-tinystories-dataset)
* [Running Dataset Translation on TinyStories json](#running-dataset-translation-on-tinystories-json)
  * [Usage](#usage)
  * [Input JSON Format](#input-json-format)
  * [Output JSON Format](#output-json-format)
* [Running Mistral 7B](#running-mistral-7b)
  * [Install Steps](#install-steps)

## Download TinyStories Dataset

To download the tinystories dataset run the following script:
```bash
bash get_ts_dataset.sh
```

The above script will create the following directory and download and extract files:
```
datasets
├── archives
│   └── TinyStories_all_data.tar.gz
├── json_stories
│   ├── data00.json
│   ├── data01.json
│   ├── data02.json
│   ├── ...
│   └── data49.json
└── txt_stories
    ├── TinyStoriesV2-GPT4-train.txt
    └── TinyStoriesV2-GPT4-valid.txt
```

## Running Dataset Translation on TinyStories json

The stories and summaries can be output in plain text format or as a JSON file.
The stories can also optionally be translated to another language as well.

### Usage
```
aug_translation.py [-h] [-i INPUT] [-o OUTPUT] [-j] [-t] [-l {es,zh,ja}]
```

**Required arguments:**

- `-i INPUT`, `--input INPUT` - Input JSON file containing stories and summaries

**Optional arguments:**

- `-o OUTPUT`, `--output OUTPUT` - Output JSON file for prompts
- `-j`, `--json` - Output stories and summaries in JSON format
- `-t`, `--translate` - Enable translation of stories
- `-l {es,zh,ja}`, `--to_code {es,zh,ja}` - Translate to language code

**Examples:**

- Generate text and print to stdout:

```
aug_translation.py -i data00.json
```

- Generates JSON and saves to specified file:

```
aug_translation.py -i data00.json -o data00_en.json -j
```

- Translate stories to Spanish, output JSON format:

```
aug_translation.py -i data00.json -o data00_es.json -j -t -l es
```

### Input JSON Format

The input JSON file should contain the stories and summaries, for example:

```json
[
  {
    "story": "Story text...",
    "summary": "Summary text..."
  },
  ...
]
```

### Output JSON Format

When using `-j` option, the output will be a JSON array with each prompt as an object:

```json
[
  {
    "story": "Translated or original story",
    "summary": "Original summary",
    "language": "es"
  },
  {
    "story": "...",
    "summary": "...",
    "language": "es"
  },
  ...
]
```

## Running Mistral 7B

Scripts here download and test the Mistral 7B model with a python wrapper for
`llama.cpp` called `llama-cpp-python`.

This can be used with prompts on our data for custom augmentation.

### Install Steps

1. First install the nanogpt requirements (see main [README.md](../README.md))
2. Second install `llama-cpp-python` and dependencies via the installation
   script provided in the repo root directory:

```bash
bash install_llama_cpp_python.sh
```
3. Finally cd into this directory and run the `download_and_test_mistral.sh`
   script via sourcing (b/c will need one's python environment):

```bash
source download_and_test_mistral7b.sh
```

This script will download mistral7b if not already in the `./models` directory,
and start the `llama-cpp-python_example.py` script.

This will should complete fairly quickly with GPU acceleration.
