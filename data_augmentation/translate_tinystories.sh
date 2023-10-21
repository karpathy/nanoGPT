#!/bin/bash

# Get CPU cores
cores=$(nproc)

# Languages
declare -A langs=(
  [ar]="Arabic"
  [az]="Azerbaijani"
  [ca]="Catalan"
  [cs]="Czech"
  [da]="Danish"
  [nl]="Dutch"
  [eo]="Esperanto"
  [fi]="Finnish"
  [fr]="French"
  [de]="German"
  [el]="Greek"
  [he]="Hebrew"
  [hi]="Hindi"
  [hu]="Hungarian"
  [id]="Indonesian"
  [ga]="Irish"
  [it]="Italian"
  [ja]="Japanese"
  [ko]="Korean"
  [fa]="Persian"
  [pt]="Portuguese"
  [ru]="Russian"
  [sk]="Slovak"
  [es]="Spanish"
  [sv]="Swedish"
  [th]="Thai"
  [tr]="Turkish"
  [uk]="Ukrainian"
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
