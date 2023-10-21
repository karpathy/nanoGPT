#!/bin/bash

# Get CPU cores
cores=$(nproc)
processes=0

# Languages
declare -A langs=(
  [en]="English"
)

# Loop languages
for lang in "${!langs[@]}"
do
  # Make sure there is an output directory
  output_dir="./datasets/json_stories/${lang}"
  if [ ! -d "${output_dir}" ]; then
    mkdir -p "${output_dir}"
  fi

  # Loop datasets
  for i in {00..49}
  do
    input="./datasets/json_stories/data${i}.json"
    output="./datasets/json_stories/${lang}/data${i}_${lang}.json"

    # Check if already translated
    if [ ! -f "${output}" ]; then

      if [ "$processes" -ge 20 ]; then
        wait
        processes=0
        echo "processes cleared at $processes"
      fi

      # Construct command
      ./venv/bin/python3 aug_translation.py -i ${input} -o ${output} -j -l ${lang}&
      processes=$((processes+1))
      echo "number processes is $processes"

    fi
  done

  # Run parallel
done

wait
