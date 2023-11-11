# Data-Shuffler for ML Permutation Invariance

These Python scripts process time-series data from a CSV file to add permutation
invariance to the csv's data fields.

Each row in the CSV becomes a single line in the text file, with each cell
represented by a unique lowercase letter (starting from 'a') followed by the
value from the cell.

One has the option to shuffle the letter-value pairs in each line, using a
command-line flag.

Training on this data with the shuffle option, will create a form of
in-frame-permutation invariance.

This will give -- during inference -- data the freedom to move around and unlock
special capabilities otherwise not available to fixed-frame trained networks.

For example, one can utilize a beam search for each of the labels to determine
which of the letter value pairs gives the strongest certainty of data points in
this frame, and build the next frame up incrementally using this technique.

# TOC

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
* [Usage](#usage)
  * [Arguments](#arguments)
  * [Example](#example)

## Getting Started

### Prerequisites

- Python (3.6 or above)

## Usage

1. Separate your file into one with timestamp columns and one with data columns.

2. Navigate to the directory where the script is located and run process_csv on
   one's data-column file:

```sh
python3 process_csv.py <data_column_file> <processed_data_file> --shuffle --exclude e
```

3. Recombine the output file from process_csv.py with the time column data.


```sh
python3 combine_csvs.py <time_column_file> <processed_data_file> <processed_csv>
```

4. Prepare the processed_data_file for training

```sh
python3 prepare.py -i <processed_data_file>
```

5. `cd` to the `explorations` folder, and utilize the script to run training:


```sh
cd ../../explorations
bash run_csv_data_training.sh
```


6. After running training, use the `unprocess.py` script to prepare the data for
   graphing.

```bash
python3 unprocess.py -i <processed_data_file> -o forecast.csv --convert_to_csv
```

7. [Optional] Create an exploration script to test training and inference with
   and without with and without shuffling.

### Arguments

- `input_file`: The path to the input CSV file containing time-series data.
- `output_file`: The path to the output text file.
- `--shuffle`: Optional flag to shuffle the order of letter-value pairs in each line.
- `--exclude`: Optional flag to remove any letters used by the dataset (e.g. `e`
    for scientific notation)

### Example

For a full example see the `main.sh` script on generated sine + noise data.

## License

This project is licensed under the MIT License

