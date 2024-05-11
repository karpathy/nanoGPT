import pandas as pd
import requests
import os
import argparse
import json
from tqdm import tqdm


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


def convert_to_json(parquet_path, json_path):
    """
    Convert Parquet file to JSON.
    """
    if not os.path.exists(json_path):
        df = pd.read_parquet(parquet_path)
        df.to_json(json_path, orient="records")
        print(f"Converted {parquet_path} to JSON at {json_path}")
    else:
        print(f"{json_path} already exists, continuing")


def emit_json_contents(json_path, output_text_file, lang="eng", filter=None):
    """
    Emit the contents of the JSON file
    Optionally, write the output to a text file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    with open(output_text_file, "a") as f:
        for item in data:
            if item["language_code"] == lang:
                content_line = f"{item['inputs']}"
                if filter != None and filter in content_line:
                    continue
                f.write(content_line.strip())
                f.write("\n")  # Separator between items
                content_line = f"{item['targets']}"
                if filter != None and filter in content_line:
                    continue
                f.write(content_line.strip())
                f.write("\n\n")  # Separator between items


def main(output_text_file, lang, filter):
    parquet_files = {
        "test_00000-of-00001": "https://huggingface.co/datasets/CohereForAI/aya_dataset/resolve/main/data/test-00000-of-00001.parquet?download=true",
        "train_00000-of-00001": "https://huggingface.co/datasets/CohereForAI/aya_dataset/resolve/main/data/train-00000-of-00001.parquet?download=true",
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

        # Download the Parquet file if it doesn't already exist
        if not os.path.exists(parquet_path):
            download_file(url, parquet_path)

        # Convert the Parquet file to JSON
        convert_to_json(parquet_path, json_path)

        # Emit the JSON contents and write output to a text file
        emit_json_contents(json_path, output_text_file, lang=lang, filter=filter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Parquet files to JSON save its contents to a text file."
    )

    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default="input.txt",
        help="Path to the output text file where the contents should be saved.",
    )

    parser.add_argument(
        "-l",
        "--lang",
        help="three letter language code e.g. 'eng', 'jpn'",
    )
    parser.add_argument(
        "-f",
        "--filter",
        help="not all language_codes are accurate, adding this to help further filter entries that contain certain keywords, e.g. 'Apakah' for the 'eng' language_code",
    )

    args = parser.parse_args()
    main(args.output_text_file, args.lang, args.filter)
