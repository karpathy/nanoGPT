import argparse
import os
import torch
import csv
import re
from rich.console import Console
from rich.table import Table

def get_best_val_loss_and_iter_num(checkpoint_file):
    """
    Extracts the best validation loss and the corresponding iteration number from a PyTorch checkpoint file.

    Args:
        checkpoint_file (str): Path to the PyTorch checkpoint file.

    Returns:
        float: The best validation loss.
        int: The iteration number corresponding to the best validation loss.
    """
    # Load the checkpoint on CPU
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

    best_val_loss = checkpoint['best_val_loss']
    iter_num = checkpoint['iter_num']

    return best_val_loss, iter_num

def find_ckpt_files(directory, path_regex=None):
    """
    Recursively finds all 'ckpt.pt' files in the given directory.

    Args:
        directory (str): The directory to search.
        path_regex (str): Regular expression to filter the checkpoint file paths.

    Returns:
        list: A list of paths to the 'ckpt.pt' files.
    """
    ckpt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('ckpt.pt'):
                ckpt_file = os.path.join(root, file)
                if path_regex is None or re.search(path_regex, ckpt_file):
                    ckpt_files.append(ckpt_file)
    return ckpt_files

def get_short_ckpt_file(ckpt_file):
    """
    Removes the '/ckpt.pt' suffix from the checkpoint file path.

    Args:
        ckpt_file (str): The full checkpoint file path.

    Returns:
        str: The checkpoint file path with the '/ckpt.pt' suffix removed.
    """
    if ckpt_file.endswith('/ckpt.pt'):
        return ckpt_file[:-8]
    else:
        return ckpt_file

def main():
    parser = argparse.ArgumentParser(description='Extract best validation loss and iteration number from PyTorch checkpoint files.')
    parser.add_argument('--directory', type=str, help='Path to the directory containing the checkpoint files.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file containing the checkpoint data.')
    parser.add_argument('--path_regex', type=str, help='Regular expression to filter the checkpoint file paths.')
    parser.add_argument('--sort', type=str, choices=['path', 'loss', 'iter'], default='path', help='Sort the table by checkpoint file path, best validation loss, or iteration number.')
    parser.add_argument('--reverse', action='store_true', help='Reverse the sort order.')
    parser.add_argument('--output', type=str, help='Path to the output CSV file.')
    args = parser.parse_args()

    if args.directory:
        ckpt_files = find_ckpt_files(args.directory, args.path_regex)

        # Extract the best validation loss and iteration number for each checkpoint file
        ckpt_data = [(get_short_ckpt_file(ckpt_file), *get_best_val_loss_and_iter_num(ckpt_file)) for ckpt_file in ckpt_files]
    elif args.csv_file:
        ckpt_data = []
        with open(args.csv_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                ckpt_data.append((get_short_ckpt_file(row[0]), float(row[1]), int(row[2])))
    else:
        print("Please provide either a directory or a CSV file.")
        return

    # Sort the data based on the specified sort option
    if args.sort == 'path':
        ckpt_data.sort(key=lambda x: x[0], reverse=args.reverse)
    elif args.sort == 'loss':
        ckpt_data.sort(key=lambda x: x[1], reverse=args.reverse)
    elif args.sort == 'iter':
        ckpt_data.sort(key=lambda x: x[2], reverse=args.reverse)

    console = None
    # Check if the TERM environment variable is set to a value that supports ANSI escape codes
    if 'TERM' in os.environ and os.environ['TERM'] in ['xterm', 'xterm-color', 'xterm-256color', 'screen', 'screen-256color', 'tmux', 'tmux-256color']:
        console = Console(color_system="standard")
    else:
        console = Console()

    # Determine the maximum length of the checkpoint file paths
    max_path_length = max(len(ckpt_file) for ckpt_file, _, _ in ckpt_data)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Checkpoint File", style="dim", width=max_path_length + 2)
    table.add_column("Best Validation Loss", justify="right")
    table.add_column("Iteration Number", justify="right")

    if args.output:
        with open(args.output, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Checkpoint File", "Best Validation Loss", "Iteration Number"])
            for ckpt_file, best_val_loss, iter_num in ckpt_data:
                table.add_row(ckpt_file, f"{best_val_loss:.4f}", str(iter_num))
                csv_writer.writerow([ckpt_file, f"{best_val_loss:.4f}", str(iter_num)])
            print(f"Results exported to {args.output}")
    else:
        for ckpt_file, best_val_loss, iter_num in ckpt_data:
            table.add_row(ckpt_file, f"{best_val_loss:.4f}", str(iter_num))

    console.print(table)

if __name__ == "__main__":
    main()
