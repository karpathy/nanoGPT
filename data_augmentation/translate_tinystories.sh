#!/bin/bash

# Get CPU cores
cores=$(nproc)

# Languages
declare -A langs=(
  [en]="English"
  [ar]="Arabic"
  [zh]="Chinese"
  [nl]="Dutch"
  [fr]="French"
  [de]="German"
  [hi]="Hindi"
  [id]="Indonesian"
  [ga]="Irish"
  [it]="Italian"
  [ja]="Japanese"
  [ko]="Korean"
  [pl]="Polish"
  [pt]="Portuguese"
  [ru]="Russian"
  [es]="Spanish"
  [tr]="Turkish"
  [uk]="Ukrainian"
  [vi]="Vietnamese"
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

      # Construct command
      cmds+="python3 aug_translation.py -i ${input} -o ${output} -j -t -l ${lang} ; "

    fi
  done

  # Run parallel
  parallel --jobs "$cores" -N"$(($cores*2))" --lb "$cmds"
done
