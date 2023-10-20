#!/bin/bash
# Get TinyStories datasets

# Directories
datasets_root_folder="datasets"
json_folder="json_stories"
txt_folder="txt_stories"
archive_folder="archives"

# Create necessary directory structure
if [[ ! -d "${datasets_root_folder}" ]]; then
  mkdir -p "${datasets_root_folder}"
fi

if [[ ! -d "${json_folder}" ]]; then
  mkdir -p "${datasets_root_folder}/${json_folder}"
fi

if [[ ! -d "${txt_folder}" ]]; then
  mkdir -p "${datasets_root_folder}/${txt_folder}"
fi

if [[ ! -d "${archive_folder}" ]]; then
  mkdir -p "${datasets_root_folder}/${archive_folder}"
fi

# Obtain the files if they don't already exist
wget -nc -P "${datasets_root_folder}/${txt_folder}" https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget -nc -P "${datasets_root_folder}/${txt_folder}" https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget -nc -P "${datasets_root_folder}/${archive_folder}" https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz

if [ -f  "${datasets_root_folder}/${json_folder}/data49.json" ]; then
  exit 0
else
  # subshell for the json extraction
  (
  cd "${datasets_root_folder}/${archive_folder}"
  tar xvfz TinyStories_all_data.tar.gz -C "../${json_folder}"
  )
fi
