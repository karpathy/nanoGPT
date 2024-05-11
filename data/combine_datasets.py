import numpy as np
import argparse
import os

def combine_binaries(file_paths, output_file):
    """Combine binary files from provided paths into a single file."""
    combined_data = []
    for file_path in file_paths:
        data = np.fromfile(file_path, dtype=np.uint16)  # Adjust dtype if needed
        combined_data.append(data)
    combined_data = np.concatenate(combined_data)
    combined_data.tofile(output_file)
    print(f"Combined {len(file_paths)} files into {output_file} with {len(combined_data)} entries.")

def main():
    parser = argparse.ArgumentParser(
        description="Combine binary files from multiple directories into single files in a specified output directory."
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs='+',
        required=True,
        help="List of directories containing train.bin and val.bin files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the combined train and validation binary data files",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate file paths for train and validation binaries
    train_files = [os.path.join(dir, "train.bin") for dir in args.dirs]
    val_files = [os.path.join(dir, "val.bin") for dir in args.dirs]

    # Specify output files for combined binaries
    output_train_file = os.path.join(args.output_dir, "train.bin")
    output_val_file = os.path.join(args.output_dir, "val.bin")

    # Combine train and validation binaries
    combine_binaries(train_files, output_train_file)
    combine_binaries(val_files, output_val_file)

if __name__ == "__main__":
    main()

