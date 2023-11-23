#!/bin/bash
# This is an example using generated data to demonstrate:
# 1. labelling of different time series data
# 2. shuffling of different fields with their label

set -x

# CREATE SYNTHETIC DATA -- Skip if using your own data
# This is a generator create two csvs (if `--split` is specified):
# 1. time data called:    `time_filename.csv`
# 2. signal data called:  `data_filename.csv`
# Note: for training may want to set the modulo < points to help with timeseries forecast
python3 sine_noise_generator.py --noise_level 0.3 --filename sine_data.csv --scientific --precision 2 --modulo 10000 --points 1000 --split

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
python3 combine_csvs.py -l time_sine_data.csv -r sine_noise_sn_shuffled.csv -o processed_sine_data.csv

set +x
echo -e "\nPreview: Timestamps with Shuffled Data -- Use This For Training"
head processed_sine_data.csv
set -x

# unprocess
python3 unprocess.py -i sine_noise_sn_shuffled.csv -o sine_data_unshuffled.csv -l abc -c

set +x
echo -e "\nPreview: Undo shuffling Of Data"
head sine_data_unshuffled.csv
set -x

# recombine time and unshuffled data
python3 combine_csvs.py -l time_sine_data.csv -r sine_data_unshuffled.csv -o sine_time_data_unshuffled_recombined.csv -d","

set +x
echo -e "\nPreview: recombined data"
head sine_time_data_unshuffled_recombined.csv
set -x

results_plot_dir="./results/"
if [ ! -d "${results_plot_dir}" ]; then
  mkdir -p "${results_plot_dir}"
fi

cp sine_time_data_unshuffled_recombined.csv "${results_plot_dir}"

set +x
echo -e "\nProcess overview complete: use 'python3 app.py results' to view recombined graph"
set -x
