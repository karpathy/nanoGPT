#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.

# Run the Python script with the specified arguments

# Add url with dataset here:
base_url="https://huggingface.co/datasets/Zyphra/Zyda/tree/main/data/zyda_no_starcoder/"
# starcoder_base_url="https://huggingface.co/datasets/Zyphra/Zyda/tree/main/data/zyda_starcoder/"

# Zyda without starcoder -- may take around 3TB of space
datasets=(
    "zyda_arxiv"
    "zyda_c4-en"
    "zyda_peS2o"
    "zyda_pile-uncopyrighted"
    "zyda_refinedweb"
    "zyda_slimpajama"
)

# Uncomment for Starcoder
# starcoder_datasets=(
#     "zyda_starcoder-git-commits-cleaned"
#     "zyda_starcoder-github-issues-filtered-structured"
#     "zyda_starcoder-jupyter-structured-clean-dedup"
#     "zyda_starcoder-languages"
# )

for dataset in "${datasets[@]}" ; do
  echo "$dataset"
  python3 ./utils/get_parquet_dataset.py \
    --url "${base_url}${dataset}" \
    --include_keys "text" \
    --value_prefix "" \
    --append
done

# Uncomment for starcoder
# for dataset in "${starcoder_datasets[@]}" ; do
#   echo "$dataset"
#   python3 ./utils/get_parquet_dataset.py \
#     --url "${starcoder_base_url}${dataset}" \
#     --include_keys "text" \
#     --value_prefix "" \
#     --append
# done

