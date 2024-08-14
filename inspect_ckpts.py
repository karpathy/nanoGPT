import argparse
import os
import torch
import csv
import re
import time
from rich.console import Console
from rich.table import Table

def get_best_val_loss_and_iter_num(target_file, args, max_retries=5, retry_interval=2):
    """
    Extracts the best validation loss and the corresponding iteration number from a PyTorch checkpoint file,
    retrying if the file is incomplete or corrupted.

    Args:
        checkpoint_file (str): Path to the PyTorch checkpoint file.
        max_retries (int): Maximum number of retries if loading fails.
        retry_interval (int): Time (in seconds) to wait before retrying.

    Returns:
        float: The best validation loss.
        int: The iteration number corresponding to the best validation loss.
    """
    best_val_loss = "No Data"
    iter_num = "No Data"

    if args.fast:
        if os.path.exists(target_file):
            with open(target_file, "r") as file:
                try:
                    line = file.readline().strip().split(",")
                    best_val_loss = float(line[0])
                    iter_num = int(line[1])
                except ValueError:
                    print("val_loss file not found")
        training_nan = "No Data"
        training_nan_iter = "No Data"
        return best_val_loss, iter_num, training_nan, training_nan_iter
    else:
        attempts = 0
        while attempts < max_retries:
            try:
                # Load the checkpoint on CPU
                checkpoint = torch.load(target_file, map_location=torch.device('cpu'))
                best_val_loss = checkpoint['best_val_loss']
                iter_num = checkpoint['iter_num']

                training_nan = None
                training_nan_iter = None
                if args.inspect_nan:
                    if 'nan' in checkpoint:
                        training_nan = checkpoint['nan']
                        training_nan_iter = checkpoint['nan_iter_num']
                    else:
                        training_nan = "No Data"
                        training_nan_iter = "No Data"

                return best_val_loss, iter_num, training_nan, training_nan_iter
            except RuntimeError as e:
                attempts += 1
                time.sleep(retry_interval)

        # If all retries fail, return "No Data"
        training_nan = "No Data"
        training_nan_iter = "No Data"
        return best_val_loss, iter_num, training_nan, training_nan_iter

def find_target_files(directory, target_string="ckpt.pt", path_regex=None):
    """
    Recursively finds all files in the given directory matching 'target string'.

    Args:
        directory (str): The directory to search.
        path_regex (str): Regular expression to filter the checkpoint file paths.

    Returns:
        list: A list of paths to target files.
    """
    ckpt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(target_string):
                ckpt_file = os.path.join(root, file)
                if path_regex is None or re.search(path_regex, ckpt_file):
                    ckpt_files.append(ckpt_file)
    return ckpt_files

def get_shortname_target_file(ckpt_file, n_fields=None, target_string="ckpt.pt"):
    """
    Extracts the last n fields (separated by hyphens) from the checkpoint file path.

    Args:
        ckpt_file (str): The full checkpoint file path.
        n_fields (int): The number of fields to display from the end of the file path.

    Returns:
        str: The shortened checkpoint file path with the last n fields.
    """
    if ckpt_file.endswith(target_string):
        ckpt_file = ckpt_file[:-(len(target_string) + 1)]
    if n_fields is not None:
        fields = ckpt_file.split('-')
        if len(fields) > n_fields:
            return '-'.join(fields[-n_fields:])
    return ckpt_file

def load_checkpoint_data(args):
    """
    Load checkpoint data from either a directory or a CSV file.

    Args:
        args (Namespace): Parsed command-line arguments.

    Returns:
        list: A list of tuples containing checkpoint data.
    """
    if args.csv_file:
        ckpt_data = []
        with open(args.csv_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            if args.inspect_nan:
                for row in csv_reader:
                    if args.path_regex is None or re.search(args.path_regex, row[0]):
                        ckpt_data.append((get_shortname_target_file(row[0]),
                                          float(row[1]),
                                          int(row[2]),
                                          str(row[3]),
                                          str(row[4])
                                          ))
            else:
                for row in csv_reader:
                    ckpt_data.append((get_shortname_target_file(row[0]), float(row[1]), int(row[2])))
    elif args.fast:
        target_files = find_target_files(args.directory, target_string="best_val_loss_and_iter.txt", path_regex=args.path_regex)
        ckpt_data = [(get_shortname_target_file(target_file, target_string="best_val_loss_and_iter.txt"),
                      *get_best_val_loss_and_iter_num(target_file, args)) for target_file in target_files]
    elif args.directory:
        ckpt_files = find_target_files(args.directory, target_string="ckpt.pt", path_regex=args.path_regex)
        ckpt_data = [(get_shortname_target_file(ckpt_file),
                      *get_best_val_loss_and_iter_num(ckpt_file, args)) for ckpt_file in ckpt_files]
    else:
        print("Please provide either a directory or a CSV file.")
        return []

    return ckpt_data

def sort_checkpoint_data(ckpt_data, sort_key, reverse):
    """
    Sort the checkpoint data based on the specified sort key.

    Args:
        ckpt_data (list): The checkpoint data to sort.
        sort_key (str): The key to sort by ('path', 'loss', 'iter', 'nan', 'nan_iter').
        reverse (bool): Whether to reverse the sort order.

    Returns:
        list: The sorted checkpoint data.
    """
    sort_keys = {
        'path': lambda x: x[0],
        'loss': lambda x: x[1],
        'iter': lambda x: x[2],
        'nan': lambda x: x[3],
        'nan_iter': lambda x: x[4]
    }
    return sorted(ckpt_data, key=sort_keys[sort_key], reverse=reverse)

def display_checkpoint_data(ckpt_data, args):
    """
    Display the checkpoint data in a table format using Rich library.

    Args:
        ckpt_data (list): The checkpoint data to display.
        args (Namespace): Parsed command-line arguments.
    """
    console = Console(color_system="standard")
    max_path_length = max(len(ckpt_file) for ckpt_file, _, _, _, _ in ckpt_data)

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Ckpt File", style="", width=max_path_length + 2)
    table.add_column("Best Val Loss", justify="right")
    table.add_column("Iter Num", justify="right")
    if args.inspect_nan:
        table.add_column("NaN Result", justify="right")
        table.add_column("NaN Iter Num", justify="right")

    for ckpt_file, best_val_loss, iter_num, training_nan, training_nan_iter in ckpt_data:
        row = [ckpt_file, f"{best_val_loss:.4f}", str(iter_num)]
        if args.inspect_nan:
            row.extend([str(training_nan), str(training_nan_iter)])
        table.add_row(*row)

    console.print(table)

def export_checkpoint_data_to_csv(ckpt_data, output_path, args):
    """
    Export the checkpoint data to a CSV file.

    Args:
        ckpt_data (list): The checkpoint data to export.
        output_path (str): The path to the output CSV file.
        args (Namespace): Parsed command-line arguments.
    """
    with open(output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = ["Checkpoint File", "Best Validation Loss", "Iteration Number"]
        if args.inspect_nan:
            headers.extend(["NaN", "Nan Iter"])
        csv_writer.writerow(headers)

        for ckpt_file, best_val_loss, iter_num, training_nan, training_nan_iter in ckpt_data:
            row = [ckpt_file, f"{best_val_loss:.4f}", str(iter_num)]
            if args.inspect_nan:
                row.extend([str(training_nan), str(training_nan_iter)])
            csv_writer.writerow(row)

    print(f"Results exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract best validation loss and iteration number from PyTorch checkpoint files.')
    parser.add_argument('--inspect_nan', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--directory', type=str, default=".", help='Path to the directory containing the checkpoint files.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file containing the checkpoint data.')
    parser.add_argument('--path_regex', type=str, help='Regular expression to filter the checkpoint file paths.')
    parser.add_argument('--sort', type=str, choices=['path', 'loss', 'iter', 'nan', 'nan_iter'], default='path', help='Sort the table by checkpoint file path, best validation loss, or iteration number.')
    parser.add_argument('--reverse', action='store_true', help='Reverse the sort order.')
    parser.add_argument('--output', type=str, help='Path to the output CSV file.')
    parser.add_argument('--n_fields', type=int, help='Number of fields to display from the end of the checkpoint file path.')
    parser.add_argument('--fast', action='store_true', help='only look for validation loss files from out_dirs')
    args = parser.parse_args()

    ckpt_data = load_checkpoint_data(args)
    if not ckpt_data:
        return

    ckpt_data = sort_checkpoint_data(ckpt_data, args.sort, args.reverse)

    if args.output:
        export_checkpoint_data_to_csv(ckpt_data, args.output, args)
    else:
        display_checkpoint_data(ckpt_data, args)

if __name__ == "__main__":
    main()

