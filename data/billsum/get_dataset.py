import pandas as pd
import requests
import os
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


def convert_to_json(parquet_path, json_path):
    """
    Convert Parquet file to JSON.
    """
    df = pd.read_parquet(parquet_path)
    df.to_json(json_path, orient="records")
    print(f"Converted {parquet_path} to JSON at {json_path}")


def emit_json_contents(json_path, order, output_text_file):
    """
    Emit the contents of the JSON file according to a specified order.
    Optionally, write the output to a text file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    with open(output_text_file, "a") as f:
        for item in data:
            for field in order:
                if field in item:
                    prefix = ""
                    if field == "title":
                        prefix = "\ntitle: "
                    elif field == "text":
                        prefix = "\ntext: "
                    elif field == "summary":
                        prefix = "\nsummary: "
                    content_line = f"{prefix}{item[field]}"
                    f.write(content_line + "\n")
            f.write("\n" + "-" * 80 + "\n")  # Separator between items


def main(order, output_text_file):
    parquet_files = {
        "ca_test": "https://huggingface.co/datasets/billsum/resolve/main/data/ca_test-00000-of-00001.parquet?download=true",
        "test": "https://huggingface.co/datasets/billsum/resolve/main/data/test-00000-of-00001.parquet?download=true",
        "train": "https://huggingface.co/datasets/billsum/resolve/main/data/train-00000-of-00001.parquet?download=true",
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

        # Download the Parquet file
        download_file(url, parquet_path)

        # Convert the Parquet file to JSON
        convert_to_json(parquet_path, json_path)

        # Emit the JSON contents and optionally write to a text file
        emit_json_contents(json_path, order, output_text_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Parquet files to JSON and emit or save its contents."
    )

    parser.add_argument(
        "--order",
        nargs="+",
        default=["title", "text", "summary"],
        help="Order of fields to emit for each record.",
    )
    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default="input.txt",
        help="Path to the output text file where the contents should be saved. If not specified, the contents will only be printed.",
    )

    args = parser.parse_args()
    main(args.order, args.output_text_file)
