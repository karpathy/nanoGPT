#!/bin/bash
# This is an example using generated data to demonstrate:
# 1. labelling of different time series data
# 2. shuffling of different fields with their label

set -x

# 1. time data is called:    `time_csv`
# 2. signal data is called:  `data_csv`

time_csv="$1"
data_csv="$2"
exclude_labels="$3"
data_shuffled_txt="${2}.shuffled.txt"
processed_txt="${2}.processed.txt"

set +x
echo -e "\nPreview: Times"
head "$time_csv"

echo -e "\n\nPreview: Data"
head "$data_csv"
echo -e "\n\n"
set -x

# Utilize a dataset only csv (no timestamps) in this case `data_sine_data.csv`
# This script does two things:
# 1. _prepend_ labels to the data
# 2. (optionally) shuffle the data
# Also 'e' is used for scientific notation, skip this letter when doing labelling
python3 process_csv.py "$data_csv" "$data_shuffled_txt" --shuffle --exclude "$exclude_labels"

# preview the result
set +x
echo -e "\nPreview: Shuffled Data"
head "$data_shuffled_txt"
echo -e "\n\n"
set -x

# recombine
python3 combine_csvs.py "$time_csv" "$data_shuffled_txt" "$processed_txt"

set +x
echo -e "\nPreview: Timestamps with Shuffled Data"
head "$processed_txt"

