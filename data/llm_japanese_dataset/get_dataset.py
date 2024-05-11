import pandas as pd
import requests
import os
import argparse
import json
from tqdm import tqdm
from zipfile import ZipFile

def download_file(url, filename):
    """
    Download a file from a given URL with a progress bar.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the download was successful.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Failed to download the file completely.")
    else:
        print(f"Downloaded {filename}")

def unzip_file(zip_path, extract_to):
    """
    Unzip a zip file to a specified directory.
    """
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def process_jsonl(jsonl_path, output_text_file):
    """
    Process a JSONL file and extract specific fields.
    Optionally, write the output to a text file.
    """
    with open(jsonl_path, 'r') as json_file, open(output_text_file, 'a') as f:
        for line in json_file:
            item = json.loads(line)
            content_line = f"{item['instruction']}"
            f.write(content_line.strip())
            f.write("\n")  # Separator between items
            content_line = f"{item['input']}"
            f.write(content_line.strip())
            f.write("\n")  # Separator between items
            content_line = f"{item['output']}"
            f.write(content_line.strip())
            f.write("\n\n")  # Separator between items

def main(output_text_file):
    zip_file_url = "https://github.com/masanorihirano/llm-japanese-dataset/releases/download/1.0.3/release-1.0.3-cc-by-sa.zip"
    zip_file_name = "./release-1.0.3-cc-by-sa.zip"
    extract_to = "./"
    jsonl_file_name = "release-1.0.3-cc-by-sa/data-cc-by-sa.jsonl"

    # Ensure the output directory exists
    os.makedirs(extract_to, exist_ok=True)

    # Download and unzip the dataset
    download_file(zip_file_url, zip_file_name)
    unzip_file(zip_file_name, extract_to)

    # Process the JSONL file
    process_jsonl(jsonl_file_name, output_text_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process a JSONL file from a zip archive."
    )

    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default="processed_output.txt",
        help="Path to the output text file where the contents should be saved.",
    )

    args = parser.parse_args()
    main(args.output_text_file)

