#!/bin/bash

set +x

download_dir="./downloaded_jsons"
output_file="input.txt"

for (( i = 1878; i < 1977; i++ )); do
  url="https://huggingface.co/datasets/dell-research-harvard/newswire/resolve/main/${i}_data_clean.json?download=true"
  if  [ ! -f "${download_dir}/${i}.json" ]; then
    wget -O "${download_dir}/${i}.json" -N "${url}" 
  else
    echo "${download_dir}/${i}.json already exists. Skipping download." 
  fi


  article_prefix=$'\n#U: Here is an article, give me the topic and the year\n'
  ca_topic_prefix=$'\n#B:\nThe topic is '
  year_prefix=$'\nThe year is '

  # Extract and prefix the "article", "ca_topic", and "year" sections and append to output file
  jq -r --arg article_prefix "$article_prefix" --arg ca_topic_prefix "$ca_topic_prefix" --arg year_prefix "$year_prefix" '
      .[] | 
      {
          article: ($article_prefix + .article),
          ca_topic: ($ca_topic_prefix + .ca_topic),
          year: ($year_prefix + (.dates[0] // ""))
      } | 
      to_entries[] | .value
    ' "${download_dir}/${i}.json" >> "$output_file"

done
