# Data-Shuffler for ML Permutation Invariance

This Python script processes time-series data from a CSV file and writes it to a
text file. Each row in the CSV becomes a single line in the text file, with each
cell represented by a unique lowercase letter (starting from 'a') followed by
the value from the cell.

You have the option to shuffle the letter-value pairs in each line, using a command-line flag.

Training on this data with the shuffle option, will yield a form of in-frame-permutation
invariance.

This will give data the freedom to move around and unlock special capabilities
otherwise not available to fixed-frame trained networks.

## Getting Started

### Prerequisites

- Python (3.6 or above)

### Installing

Clone the repository or download the script to your local machine.

```sh
git clone <repository-url>
```

Or, simply create a new Python file, for example, `process_csv.py`, and copy the above Python code into this file.

## Usage

Navigate to the directory where the script is located and run the following command:

```sh
python process_csv.py <input_file> <output_file> [--shuffle]
```

### Arguments

- `input_file`: The path to the input CSV file containing time-series data.
- `output_file`: The path to the output text file.
- `--shuffle`: Optional flag to shuffle the order of letter-value pairs in each line.

### Example

To process a CSV file without shuffling the letter-value pairs:

```sh
python process_csv.py input.csv output.txt


To process a CSV file and shuffle the letter-value pairs:

```sh
python process_csv.py input.csv output.txt --shuffle
```
## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details

