import pandas as pd
import requests
import os
import argparse

def download_file(url, filename):
    """
    Download a file from a given URL.
    """
    # Check if the file already exists
    if not os.path.exists(filename):
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()  # Ensure the download was successful.
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f'Downloaded {filename}')
    else:
        print(f'File {filename} already exists, skipping download.')

def convert_to_json_and_write(parquet_path, output_text_file):
    """
    Convert Parquet file to JSON and write specified fields to an output file.
    """
    df = pd.read_parquet(parquet_path)

    # Open the output file for appending
    with open(output_text_file, 'a') as f:
        for _, row in df.iterrows():
            # Assuming 'article' and 'highlights' are columns in the DataFrame
            if 'article' in row and 'highlights' in row:
                f.write(f"text: {row['article']}\n")
                f.write(f"summary: {row['highlights']}\n")
                f.write('\n\n')  # Separator between articles

def main(output_text_file):
    files = {
        "test": "https://huggingface.co/datasets/cnn_dailymail/resolve/main/3.0.0/test-00000-of-00001.parquet?download=true",
        "train_0": "https://huggingface.co/datasets/cnn_dailymail/resolve/main/3.0.0/train-00000-of-00003.parquet?download=true",
        "train_1": "https://huggingface.co/datasets/cnn_dailymail/resolve/main/3.0.0/train-00001-of-00003.parquet?download=true",
        "train_2": "https://huggingface.co/datasets/cnn_dailymail/resolve/main/3.0.0/train-00002-of-00003.parquet?download=true",
        "validation": "https://huggingface.co/datasets/cnn_dailymail/resolve/main/3.0.0/validation-00000-of-00001.parquet?download=true"
    }

    download_dir = './downloaded_parquets'

    os.makedirs(download_dir, exist_ok=True)

    for file_name, url in files.items():
        parquet_path = os.path.join(download_dir, file_name + '.parquet')

        # Download the Parquet file only if it doesn't already exist
        download_file(url, parquet_path)

        # Convert the Parquet file and write to the specified text file
        convert_to_json_and_write(parquet_path, output_text_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Parquet files, convert to JSON, and write specific fields to a text file.")
    parser.add_argument("--output_text_file", type=str, default="input.txt",
                        help="Output text file name. Defaults to 'input.txt'.")

    args = parser.parse_args()
    main(args.output_text_file)

