import pandas as pd
import requests
import os
import argparse
import json
from tqdm import tqdm
from bs4 import BeautifulSoup


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


def emit_json_contents(
    json_path,
    output_text_file,
    include_keys,
    value_prefixes,
    required_key,
    skip_empty,
    exclude,
):
    """
    Emit the contents of the JSON file
    Optionally, write the output to a text file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    skip_item = False
    excluded_pairs = {}
    if exclude:
        for pair in args.exclude:
            for i in range(0, len(pair), 2):
                key = pair[i]
                value = pair[i+1]
                excluded_pairs[key] = value

    with open(output_text_file, "a") as f:
        prev_item_written = False
        for item in data:
            if required_key and item[required_key] == "":
                continue  # Skip entire item if required key is empty

            for key, prefix in zip(include_keys, value_prefixes):
                if key in item:
                    if skip_empty and item[key] == "":
                        continue  # skip empty items if activated
                    for i_key, i_value in item.items():
                        if (i_key in excluded_pairs) and (i_value == excluded_pairs[i_key]):
                            # print(key, item[key], "not included")
                            skip_item=True
                        if skip_item:
                            break

                    if skip_item:
                        continue

                    if prev_item_written:
                        f.write("\n")  # Single newline between items
                    content_line = f"{prefix}{item[key]}"
                    f.write(content_line.strip())
                    prev_item_written = True


def find_parquet_links(url):
    """
    Find all parquet file links on the given URL.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [
        ("https://huggingface.co" + a["href"])
        for a in soup.find_all("a", href=True)
        if a["href"].endswith(".parquet?download=true")
    ]
    return links


def main(
    url,
    output_text_file,
    include_keys,
    value_prefixes,
    required_key,
    skip_empty,
    exclude,
    append,
):
    parquet_links = find_parquet_links(url)
    download_dir = "./downloaded_parquets"
    json_dir = "./json_output"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    if not append:
        open(output_text_file, "w").close()

    for link in parquet_links:
        file_name = link.split("/")[-1].split("?")[0]  # Extract filename
        parquet_path = os.path.join(download_dir, file_name)
        json_path = os.path.join(json_dir, file_name.replace(".parquet", ".json"))

        # Download the Parquet file if it doesn't already exist
        if not os.path.exists(parquet_path):
            download_file(link, parquet_path)

        # Convert the Parquet file to JSON
        convert_to_json(parquet_path, json_path)

        # Emit the JSON contents and write output to a text file
        emit_json_contents(
            json_path,
            output_text_file,
            include_keys,
            value_prefixes,
            required_key,
            skip_empty,
            exclude,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape and convert Parquet files from URL to JSON and save its contents to a text file."
    )

    parser.add_argument(
        "--url",
        type=str,
        default="https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/tree/main/english",
        help="URL to scrape for Parquet files.",
    )
    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default="input.txt",
        help="Path to the output text file where the contents should be saved.",
    )
    parser.add_argument(
        "-i",
        "--include_keys",
        type=str,
        nargs="+",
        required=True,
        help="List of keys to include from the JSON contents.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        action="append",
        metavar=("KEY", "VALUE"),
        help="Specify key-value pairs to be excluded. Use the format: --exclude KEY VALUE [KEY VALUE ...]",
    )
    parser.add_argument(
        "-p",
        "--value_prefix",
        type=str,
        nargs="+",
        required=True,
        help="List of prefixes to be added to each individual value when emitting to the text file.",
    )
    parser.add_argument(
        "-r",
        "--required_key",
        type=str,
        default=None,
        help="Only emit items that have this key (optional).",
    )
    parser.add_argument(
        "-s",
        "--skip_empty",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Skip any item which is the empty string",
    )
    parser.add_argument(
        "-a",
        "--append",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="append to the current input.txt file",
    )
    args = parser.parse_args()
    main(
        args.url,
        args.output_text_file,
        args.include_keys,
        args.value_prefix,
        args.required_key,
        args.skip_empty,
        args.exclude,
        args.append,
    )
