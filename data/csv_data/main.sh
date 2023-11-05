#!/bin/bash

python3 sine_noise_generator.py --noise_level 0.3 --filename sine_data.csv --scientific --precision 2 --modulo 1000 --points 1000000 --split
python3 process_csv.py data_sine_data.csv sine_noise_sn_shuffled.csv --shuffle --exclude e; head sine_noise_sn_shuffled.csv
python3 combine_csvs.py time_sine_data.csv sine_noise_sn_shuffled.csv processed_sine_data.csv
