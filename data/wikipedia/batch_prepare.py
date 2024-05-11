import os
import subprocess
import argparse
import numpy as np

def batch_prepare(input_dir, train_output, val_output, prepare_script, train_ratio=0.9):
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')])
    num_train = int(len(files) * train_ratio)

    # Train files
    for file in files[:num_train]:
        subprocess.run(['python3', prepare_script, '--train_input', file, '--train_output', file.replace('.txt', '.bin'), '-p 1.0'])

    # Validation files
    for file in files[num_train:]:
        subprocess.run(['python3', prepare_script, '--train_input', file, '--train_output', file.replace('.txt', '.bin'), '-p 1.0'])

    # Combine bins
    combine_bins(files[:num_train], train_output)
    combine_bins(files[num_train:], val_output)
    print(f"Created {train_output} and {val_output}")

def combine_bins(files, output_file):
    all_data = []
    for file in files:
        bin_file = file.replace('.txt', '.bin')
        if os.path.exists(bin_file):
            all_data.append(np.fromfile(bin_file, dtype=np.uint16))  # Adjust dtype based on your tokenizer's output
    combined = np.concatenate(all_data)
    combined.tofile(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training and validation binary files from divided text files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the divided files.")
    parser.add_argument("--train_output", type=str, default="train.bin", help="Output binary file for training data.")
    parser.add_argument("--val_output", type=str, default="val.bin", help="Output binary file for validation data.")
    parser.add_argument("--prepare_script", type=str, required=True, help="Path to the prepare.py script.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of training data to total data.")

    args = parser.parse_args()
    batch_prepare(args.input_dir, args.train_output, args.val_output, args.prepare_script, args.train_ratio)

