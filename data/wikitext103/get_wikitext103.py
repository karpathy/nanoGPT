import os
import requests
import argparse
import json


def download_file(url, filename):
    """
    Download a file from a given URL.
    """
    response = requests.get(url)
    response.raise_for_status()  # Ensure the download was successful.
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {filename}")


def main(output_text_file):
    parquet_files = {
        "train-00000-of-00002": "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-v1/train-00000-of-00002.parquet?download=true",
        "train-00001-of-00002": "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-v1/train-00001-of-00002.parquet?download=true",
        "validation-00000-of-00001": "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-v1/validation-00000-of-00001.parquet?download=true",
        "test-00000-of-00001": "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-v1/test-00000-of-00001.parquet?download=true"
    }

    download_dir = "./downloaded_parquets"
    json_dir = "./json_output"

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    if output_text_file:
        # Ensure the output text file is empty before starting
        open(output_text_file, "w").close()

    for file_name, url in parquet_files.items():
        parquet_path = os.path.join(download_dir, file_name + ".parquet")
        json_path = os.path.join(json_dir, file_name + ".json")

        if not os.path.exists(parquet_path):
            # Download the Parquet file only if it doesn't exist locally
            download_file(url, parquet_path)

        # Convert Parquet file to JSON (if needed, can be skipped)
        # This step can be omitted if you're directly working with Parquet files
        # Convert the Parquet file to JSON
        # convert_to_json(parquet_path, json_path)

        with open(json_path, "r") as f:
            data = json.load(f)

        with open(output_text_file, "a") as f:
            for item in data:
                f.write(item["text"].strip() + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print text values from JSON files to a text file."
    )

    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default="input.txt",
        help="Path to the output text file where the contents should be saved."
    )

    args = parser.parse_args()
    main(args.output_text_file)

