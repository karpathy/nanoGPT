#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_PARQUET_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, or make this empty to skip
# 4. Set "skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "no_output_text" to true if you plan to process the intermediate json files in a custom manner.

### BEGIN USER MODIFIED AREA
# Modify these variables to customize your dataset extraction
url="INSERT_URL_WITH_PARQUET_FILES"
include_keys=("question" "options" "cot_content" "answers")
value_prefixes=("question: " "options: " "cot: " "answer: ") # Leave this array empty if no prefixes are needed
skip_empty=true
no_output_text=false
### END USER MODIFIED AREA

# Construct the arguments for the script
include_keys_arg=$(printf -- '--include_keys "%s" ' "${include_keys[@]}")

# Only add the value prefixes argument if the array is not empty
if [ ${#value_prefixes[@]} -ne 0 ]; then
  value_prefixes_arg=$(printf -- '--value_prefixes "%s" ' "${value_prefixes[@]}")
else
  value_prefixes_arg=""
fi

# Only add the skip_empty argument if it's set to true
if [ "$skip_empty" = true ]; then
  skip_empty_arg="--skip_empty"
else
  skip_empty_arg=""
fi

# Only add the skip_empty argument if it's set to true
if [ "$no_output_text" = true ]; then
  no_output_text_arg="--no_output_text"
else
  no_output_text_arg=""
fi

# Run the Python script with the specified arguments
python3 ./utils/get_parquet_dataset.py \
  --url "${url}" \
  ${include_keys_arg} \
  ${value_prefixes_arg} \
  ${skip_empty_arg} \
  ${no_output_text_arg}

