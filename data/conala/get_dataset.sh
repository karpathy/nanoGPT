#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_PARQUET_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.

# Modify for the url
url="https://huggingface.co/datasets/neulab/conala/tree/main/data"

# Run the Python script with the specified arguments
# Note: the $'\n' syntax allows for special characters like \n
python3 ./utils/get_json_dataset.py \
  --url "${url}" \
  --include_keys "intent" "snippet" \
  --value_prefix $'#U:\n' $'#B:\n'

# U for user 'B' for bot
