#!/bin/bash

# This creates three folders which each will have their
# own input.txt

# Python-edu
subdir="python-edu"
if [[ ! -d "$subdir" ]]; then
  mkdir "$subdir"
fi
pushd "$subdir"
url="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main/python-edu"
python3 ../utils/get_parquet_dataset.py \
  --url "${url}" \
  --include_keys "repo_name" "blob_id" "path" \
  --value_prefix $'\n' $'\n' $'\n'
popd

# TODO: add step to process and obtain python dataset

# Fineweb-edu-dedup
subdir="fineweb-edu-dedup"
if [[ ! -d "$subdir" ]]; then
  mkdir "$subdir"
fi
pushd "$subdir"
url="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main/fineweb-edu-dedup"
python3 ../utils/get_parquet_dataset.py \
  --url "${url}" \
  --include_keys "text" \
  --value_prefix $'#T:\n'
popd

# Cosmopedia-v2
subdir="cosmopedia-v2"
if [[ ! -d "$subdir" ]]; then
  mkdir "$subdir"
fi
pushd "$subdir"

url="https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/main/cosmopedia-v2"
python3 ../utils/get_parquet_dataset.py \
  --url "${url}" \
  --include_keys "prompt" "text" \
  --value_prefix $'#U:\n' $'#B:\n'
popd
