#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.

# Run the Python script with the specified arguments

url="https://huggingface.co/datasets/eastwind/tiny-sherlock-audio"
out_dir="input_mp3s"
txt_dir="texts"

if [[ ! -d "${out_dir}" ]]; then
  mkdir -p "${out_dir}"
fi

for i in $(seq -f "%02g" 1 24); do
  wget -nc -O "${out_dir}/tiny_sherlock_audio_${i}.mp3" "${url}/resolve/main/adventuresholmes_${i}_doyle_64kb.mp3?download=true"
donej

# Helps when utilizing on consumer GPUs
python3 split_mp3s.py "${out_dir}" --max_size_mb 5

# Append all files txt to the input.txt file
python3 snac_converter.py encode "split_${out_dir}" "${txt_dir}" --directory

# concatenate text files
cat "${txt_dir}"/*.txt > mp3_input.txt

# create train.bin and val.bin as well as meta.pkl
python3 prepare.py -t mp3_input.txt --numeric_range --min_token 0 --max_token 4097

