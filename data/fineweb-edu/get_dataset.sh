#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.

# if "all" specified then download everything
if [ "${1}" = "all" ]; then
  entries=(
      "CC-MAIN-2013-20"
      "CC-MAIN-2013-48"
      "CC-MAIN-2014-10"
      "CC-MAIN-2014-15"
      "CC-MAIN-2014-23"
      "CC-MAIN-2014-35"
      "CC-MAIN-2014-41"
      "CC-MAIN-2014-42"
      "CC-MAIN-2014-49"
      "CC-MAIN-2014-52"
      "CC-MAIN-2015-06"
      "CC-MAIN-2015-11"
      "CC-MAIN-2015-14"
      "CC-MAIN-2015-18"
      "CC-MAIN-2015-22"
      "CC-MAIN-2015-27"
      "CC-MAIN-2015-32"
      "CC-MAIN-2015-35"
      "CC-MAIN-2015-40"
      "CC-MAIN-2015-48"
      "CC-MAIN-2016-07"
      "CC-MAIN-2016-18"
      "CC-MAIN-2016-22"
      "CC-MAIN-2016-26"
      "CC-MAIN-2016-30"
      "CC-MAIN-2016-36"
      "CC-MAIN-2016-40"
      "CC-MAIN-2016-44"
      "CC-MAIN-2016-50"
      "CC-MAIN-2017-04"
      "CC-MAIN-2017-09"
      "CC-MAIN-2017-13"
      "CC-MAIN-2017-17"
      "CC-MAIN-2017-22"
      "CC-MAIN-2017-26"
      "CC-MAIN-2017-30"
      "CC-MAIN-2017-34"
      "CC-MAIN-2017-39"
      "CC-MAIN-2017-43"
      "CC-MAIN-2017-47"
      "CC-MAIN-2017-51"
      "CC-MAIN-2018-05"
      "CC-MAIN-2018-09"
      "CC-MAIN-2018-13"
      "CC-MAIN-2018-17"
      "CC-MAIN-2018-22"
      "CC-MAIN-2018-26"
      "CC-MAIN-2018-30"
      "CC-MAIN-2018-34"
      "CC-MAIN-2018-39"
      "CC-MAIN-2018-43"
      "CC-MAIN-2018-47"
      "CC-MAIN-2018-51"
      "CC-MAIN-2019-04"
      "CC-MAIN-2019-09"
      "CC-MAIN-2019-13"
      "CC-MAIN-2019-18"
      "CC-MAIN-2019-22"
      "CC-MAIN-2019-26"
      "CC-MAIN-2019-30"
      "CC-MAIN-2019-35"
      "CC-MAIN-2019-39"
      "CC-MAIN-2019-43"
      "CC-MAIN-2019-47"
      "CC-MAIN-2019-51"
      "CC-MAIN-2020-05"
      "CC-MAIN-2020-10"
      "CC-MAIN-2020-16"
      "CC-MAIN-2020-24"
      "CC-MAIN-2020-29"
      "CC-MAIN-2020-34"
      "CC-MAIN-2020-40"
      "CC-MAIN-2020-45"
      "CC-MAIN-2020-50"
      "CC-MAIN-2021-04"
      "CC-MAIN-2021-10"
      "CC-MAIN-2021-17"
      "CC-MAIN-2021-21"
      "CC-MAIN-2021-25"
      "CC-MAIN-2021-31"
      "CC-MAIN-2021-39"
      "CC-MAIN-2021-43"
      "CC-MAIN-2021-49"
      "CC-MAIN-2022-05"
      "CC-MAIN-2022-21"
      "CC-MAIN-2022-27"
      "CC-MAIN-2022-33"
      "CC-MAIN-2022-40"
      "CC-MAIN-2022-49"
      "CC-MAIN-2023-06"
      "CC-MAIN-2023-14"
      "CC-MAIN-2023-23"
      "CC-MAIN-2023-40"
      "CC-MAIN-2023-50"
      "CC-MAIN-2024-10"
  )

  for entry in "${entries[@]}"; do
      year=$(echo $entry | cut -d'-' -f3)
      week=$(echo $entry | cut -d'-' -f4)
      echo "Year: $year, Week: $week"
      url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/data/CC-MAIN-${year}-${week}"
      # Uncomment and fill in if url has parquet datasets
      python3 ./utils/get_parquet_dataset.py \
        --url "${url}" \
        --include_keys "text" \
        --value_prefix ''
  done

else
  # Else just get first one
  url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/data/CC-MAIN-CC-MAIN-2013-20"

  python3 ./utils/get_parquet_dataset.py \
    --url "${url}" \
    --include_keys "text" \
    --value_prefix ''
fi
