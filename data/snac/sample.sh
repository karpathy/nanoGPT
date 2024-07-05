#!/bin/bash

# Example of how to create output, assumes checkpoint createdd
pushd ../../
python3 sample.py --device=cuda --out_dir ./out --start "" --token_boundary $'\n' --num_samples 1 --sample_file ./data/snac/sample.txt

# Remove frames which do not have 7 entries.
popd
python3 format.py clean sample.txt output.txt

# Convert into mp3 format for listening
python3 snac_converter.py decode output.txt output.mp3 --input_format text

