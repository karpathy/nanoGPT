#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.

# Run the Python script with the specified arguments

# Add url with dataset here:
url="https://huggingface.co/datasets/globis-university/aozorabunko-clean/resolve/main/aozorabunko-dedupe-clean.jsonl.gz?download=true"
filename="aozorabunko_dedupe_clean.jsonl.gz"

if ([ ! -f "${filename}" ] && [ ! -f "${filename%%.gz}" ]); then
  wget -O "$filename" "$url"
fi

if [ ! -f "${filename%%.gz}" ]; then
  gunzip "$filename"
fi

# uncomment and fill in if url has json datasets
# Note: the $'\n' syntax allows for special characters like \n
python3 ./utils/get_json_dataset.py \
  --direct_json_input "${filename%%.gz}" \
  --include_keys "text" \
  --value_prefix ""
