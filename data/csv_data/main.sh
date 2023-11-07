#!/bin/bash
# This is an example using generated data to demonstrate:
# 1. labelling of different time series data
# 2. shuffling of different fields with their label

set -x

# CREATE SYNTHETIC DATA -- Skip if using your own data
# This is a generator create two csvs (if `--split` is specified):
# 1. time data called:    `time_filename.csv`
# 2. signal data called:  `data_filename.csv`
python3 sine_noise_generator.py --noise_level 0.3 --filename sine_data.csv --scientific --precision 2 --modulo 1000 --points 1000000 --split

set +x
echo -e "\nPreview: Generated Times"
head time_sine_data.csv

echo -e "\n\nPreview: Generated Data"
head data_sine_data.csv
echo -e "\n\n"
set -x

# Utilize a dataset only csv (no timestamps) in this case `data_sine_data.csv`
# This script does two things:
# 1. _prepend_ labels to the data
# 2. (optionally) shuffle the data
# Also 'e' is used for scientific notation, skip this letter when doing labelling
python3 process_csv.py data_sine_data.csv sine_noise_sn_shuffled.csv --shuffle --exclude e

# preview the result
set +x
echo -e "\nPreview: Shuffled Data"
head sine_noise_sn_shuffled.csv
echo -e "\n\n"
set -x

# recombine
python3 combine_csvs.py time_sine_data.csv sine_noise_sn_shuffled.csv processed_sine_data.csv

set +x
echo -e "\nPreview: Timestamps with Shuffled Data"
head processed_sine_data.csv
