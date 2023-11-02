#!/bin/bash
#
## Language names
declare -a lang_names=(
  Arabic
  Azerbaijani
  Catalan
  Czech
  Danish
  Dutch
  Esperanto
  Finnish
  French
  German
  Greek
  Hebrew
  Hindi
  Hungarian
  Indonesian
  Irish
  Italian
  Japanese
  Korean
  Persian
  Portuguese
  Polish
  Russian
  Slovak
  Spanish
  Swedish
  Thai
  Turkish
  Ukrainian
  Mandarin
)

# List of languages for reference
declare -a lang_codes=(
  ar # Arabic
  az # Azerbaijani
  ca # Catalan
  cs # Czech
  da # Danish
  nl # Dutch
  eo # Esperanto
  fi # Finnish
  fr # French
  de # German
  el # Greek
  he # Hebrew
  hi # Hindi
  hu # Hungarian
  id # Indonesian
  ga # Irish
  it # Italian
  ja # Japanese
  ko # Korean
  fa # Persian
  pt # Portuguese
  pl # Polish
  ru # Russian
  sk # Slovak
  es # Spanish
  sv # Swedish
  th # Thai
  tr # Turkish
  uk # Ukrainian
  zh # Mandarin
)

# Default language if none provided
lang="es"


# Print usage
function print_usage {
  echo "Usage: $0 [-l] [-h] [lang]"
  echo "Translate dataset into language code."
  echo "Options:"
  echo "  -l   List available languages"
  echo "  -h   Print this help"
  echo "If no language provided, default is Spanish (es)"
}

# Print list of available languages
function print_langs {
  echo "Available languages:"

  for ((i=0; i<${#lang_codes[@]}; i++)); do
    code=${lang_codes[$i]}
    name=${lang_names[$i]}
    echo "$code - $name"
  done
}

# Check if language code is valid
function check_lang {
  for l in "${lang_codes[@]}"; do
    if [ "$1" == "$l" ]; then
      return 0
    fi
  done

  echo "Error: Invalid language code $1" >&2
  return 1
}

# Print status update
function print_status {
  echo "$(date +"%Y-%m-%d %H:%M:%S") - $1"
}


# Parse options
while getopts ":lh" opt; do
  case $opt in
    l)
      print_langs
      exit 0
      ;;
    h)
      print_usage
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      print_usage
      exit 1
      ;;
  esac
done

# Create output directory if needed
output_dir="datasets/json_stories/$lang"
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Check if language provided
if [ $# -ge 1 ]; then
  lang="$1"
  shift

  # Check if valid
  check_lang "$lang" || exit 1
fi

# Print start message
print_status "Starting translations from en to $lang"

# Loop through data files
for i in {00..49}; do

  print_status "Translating data file $i"

  # Set file paths
  input="datasets/json_stories/archive/data${i}.json"
  output="datasets/json_stories/${lang}/data${i}_${lang}.json"

  # Print info
  echo "Input: $input"
  echo "Output: $output"

  # Translate file
  python3 aug_translation.py -i "$input" -o "$output" -j -t -l "$lang" || echo "Error translating $input" >&2

done

