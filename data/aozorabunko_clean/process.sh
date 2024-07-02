#!/bin/bash

set -x

temp_dir="temp_dir"

sed 's/ゝ//g' "$temp_dir"/direct_output.txt > "$temp_dir"/output_no_iter.txt

# Replace hiragana with romaji
python3 romaji_converter.py "$temp_dir"/output_no_iter.txt "$temp_dir"/output_cutlet.txt

sed 's/ゝ//g' "$temp_dir"/output_cutlet.txt > "$temp_dir"/output_cutlet_no_iter.txt

# Create specific character replacements
bash replace_chars.sh "$temp_dir"/output_cutlet_no_iter.txt
