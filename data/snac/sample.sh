#!/bin/bash

set -x

pushd ../../
python3 sample.py --device=cuda --out_dir ./out --start "0" --token_boundary $'\n' --num_samples 1 --sample_file ./data/snac/sample.txt --max_new_tokens 4096

# Remove frames which do not have 7 entries.
popd
python3 format.py clean sample.txt output.txt

# # Convert into mp3 format for listening
python3 snac_converter.py decode output.txt output.mp3 --input_format text

