#!/bin/bash

input_file=${1:-input.txt}
python3 ./utils/partition_file.py --input_file "${input_file}"
python3 ./utils/batch_prepare.py --input_dir partitioned_file --prepare_script prepare.py --tokenizer char
