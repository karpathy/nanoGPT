"""
Authored by Gavia Gray (https://github.com/gngdb)

Wrapper for wandb logging with efficient CSV logging and correct config JSON writing.
The CSV structure maintains a consistent order of keys based on their first appearance,
using a simple list for ordering. This ensures data integrity and allows for graceful
failure and manual recovery if needed.

Example usage:
  run = wandb.init(config=your_config)
  wrapper = LogWrapper(run, out_dir='path/to/output')

  ...
    # in train loop
    wrapper.log({"train/loss": 0.5, "train/accuracy": 0.9, "val/loss": 0.6, "val/accuracy": 0.85})
    wrapper.print("Train: {loss=:.4f}, {accuracy=:.2%}", prefix="train/")
    wrapper.print("Val: {loss=:.4f}, {accuracy=:.2%}", prefix="val/")
    wrapper.step()

  ...
    # at the end of your script
    wrapper.close()

  # If the script terminates unexpectedly, you can still recover the CSV using bash:
  # cat path/to/output/log_header.csv.tmp path/to/output/log_data.csv.tmp > path/to/output/log.csv
"""

import re
import os
import csv
import json
import atexit


def exists(x): return x is not None

def transform_format_string(s):
    """
    Transforms a string containing f-string-like expressions to a format
    compatible with str.format().

    This function converts expressions like '{var=}' or '{var=:formatting}'
    to 'var={var}' or 'var={var:formatting}' respectively. This allows
    for f-string-like syntax to be used with str.format().

    Args:
        s (str): The input string containing f-string-like expressions.

    Returns:
        str: The transformed string, compatible with str.format().

    Examples:
        >>> transform_format_string("Value is {x=}")
        "Value is x={x}"
        >>> transform_format_string("Formatted value is {x=:.2f}")
        "Formatted value is x={x:.2f}"
    """
    pattern = r'\{(\w+)=(:.[^}]*)?\}'
    return re.sub(pattern, lambda m: f"{m.group(1)}={{{m.group(1)}{m.group(2) or ''}}}", s)

class CSVLogWrapper:
    def __init__(self, logf=None, config={}, out_dir=None):
        self.logf = logf
        self.config = config
        self.log_dict = {}
        self.out_dir = out_dir
        self.csv_data_file = None
        self.csv_header_file = None
        self.csv_writer = None
        self.step_count = 0
        self.ordered_keys = []
        self.header_updated = False
        self.is_finalized = False
        self.no_sync_keyword = 'no_sync' # Keyword to prevent syncing to wandb

        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            self.setup_csv_writer()
            self.write_config()

        atexit.register(self.close)

    def setup_csv_writer(self):
        self.csv_data_path = os.path.join(self.out_dir, 'log_data.csv.tmp')
        self.csv_header_path = os.path.join(self.out_dir, 'log_header.csv.tmp')
        self.csv_data_file = open(self.csv_data_path, 'w', newline='')
        self.csv_header_file = open(self.csv_header_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_data_file)

    def write_config(self):
        if self.config:
            config_path = os.path.join(self.out_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(dict(**self.config), f, indent=2)

    def log(self, data):
        self.log_dict.update(data)
        for key in data:
            if key not in self.ordered_keys:
                self.ordered_keys.append(key)
                self.header_updated = True

    def update_header(self):
        if self.header_updated:
            header = ['step'] + self.ordered_keys
            with open(self.csv_header_path, 'w', newline='') as header_file:
                csv.writer(header_file).writerow(header)
            self.header_updated = False

    def print(self, format_string, prefix=None):
        format_string = transform_format_string(format_string)

        if prefix:
            # Filter keys with the given prefix and remove the prefix
            filtered_dict = {k.replace(prefix, ''): v for k, v in self.log_dict.items() if k.startswith(prefix)}
        else:
            filtered_dict = self.log_dict
        # replace any '/' in keys with '_'
        filtered_dict = {k.replace('/', '_'): v for k, v in filtered_dict.items()}

        try:
            print(format_string.format(**filtered_dict))
        except KeyError as e:
            print(f"KeyError: {e}. Available keys: {', '.join(filtered_dict.keys())}")
            raise e

    def step(self):
        if exists(self.logf) and self.log_dict:
            self.logf({k: v for k, v in self.log_dict.items() if self.no_sync_keyword not in k})

        if self.csv_writer and self.log_dict:
            self.update_header()

            # Prepare the row data
            row_data = [self.step_count] + [self.log_dict.get(key, '') for key in self.ordered_keys]
            self.csv_writer.writerow(row_data)
            self.csv_data_file.flush()  # Ensure data is written to file

        self.step_count += 1
        self.log_dict.clear()

    def close(self):
        if self.csv_data_file:
            self.csv_data_file.close()

        self.finalize_csv()

    def finalize_csv(self):
        if self.is_finalized:
            return

        csv_final_path = os.path.join(self.out_dir, 'log.csv')

        with open(csv_final_path, 'w', newline='') as final_csv:
            # Copy header
            with open(self.csv_header_path, 'r') as header_file:
                final_csv.write(header_file.read())

            # Copy data
            with open(self.csv_data_path, 'r') as data_file:
                final_csv.write(data_file.read())
        self.is_finalized = True

        # Remove the temporary files
        os.remove(self.csv_header_path)
        os.remove(self.csv_data_path)
