#!/bin/bash
# Get TinyStories datasets

# Directories
datasets_root_folder="datasets"
json_folder="json_stories"
txt_folder="txt_stories"

# Create necessary directory structure
if [[ ! -d "${datasets_root_folder}" ]]; then
  mkdir -p "${datasets_root_folder}"
  mkdir -p "${datasets_root_folder}"
fi

if [[ ! -d "${json_folder}" ]]; then
  mkdir -p "${datasets_root_folder}/${json_folder}"
  mkdir -p "${datasets_root_folder}/${json_folder}"
fi

if [[ ! -d "${txt_folder}" ]]; then
  mkdir -p "${datasets_root_folder}/${txt_folder}"
  mkdir -p "${datasets_root_folder}/${txt_folder}"
fi

# Obtain the files if they don't already exist
wget -nc -P "${datasets_root_folder}" https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget -nc -P "${datasets_root_folder}" https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget -nc -P "${datasets_root_folder}/${json_folder}" https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz

# subshell for the json extraction
(
cd "${datasets_root_folder}/${json_folder}"
if [ -f "data49.json" ]; then
  exit 0
else
  tar xvfz TinyStories_all_data.tar.gz
fi
)

