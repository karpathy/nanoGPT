import pandas as pd
import requests
import os
import argparse
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import subprocess

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


def emit_json_contents(json_path, output_text_file, from_lang, to_lang, phonemize):
    """
    Emit the contents of the JSON file
    Optionally, write the output to a text file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if phonemize:
        with open("from_" + output_text_file, "a") as from_file:
            with open("to_" + output_text_file, "a") as to_file:
                for item in data:
                    for _, value in item.items():
                        from_line = value[f'{from_lang}'].replace("\n", " ").strip()
                        from_file.write(from_line)
                        from_file.write("\n")  # Separator between items
                        to_line = value[f'{to_lang}'].replace("\n", " ").strip()
                        to_file.write(to_line)
                        to_file.write("\n")  # Separator between items
    else:
        with open(output_text_file, "a") as f:
            for item in data:
                for _, value in item.items():
                    from_line = value[f'{from_lang}'].replace("\n", " ").strip()
                    to_line = value[f'{to_lang}'].replace("\n", " ").strip()
                    f.write(from_line)
                    f.write("\n")  # Separator between items
                    f.write(to_line)
                    f.write("\n\n")  # Separator between items

def find_parquet_links(url):
    """
    Find all parquet file links on the given URL.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = [("https://huggingface.co" + a['href']) for a in soup.find_all('a', href=True) if a['href'].endswith('.parquet?download=true')]
    return links

def main(url, output_text_file, from_lang, to_lang, phonemize):
    parquet_links = find_parquet_links(url)

    download_dir = "./downloaded_parquets"
    json_dir = "./json_output"

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    if output_text_file:
        # Ensure the output text file is empty before starting
        open(output_text_file, "w").close()

    for link in parquet_links:
        file_name = link.split("/")[-1].split("?")[0]  # Extract filename
        parquet_path = os.path.join(download_dir, file_name)
        json_path = os.path.join(json_dir, file_name.replace('.parquet', '.json'))

        # Download the Parquet file if it doesn't already exist
        if not os.path.exists(parquet_path):
            download_file(link, parquet_path)

        # Convert the Parquet file to JSON
        convert_to_json(parquet_path, json_path)

        emit_json_contents(json_path, output_text_file, from_lang, to_lang, phonemize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape and convert Parquet files from URL to JSON and save its contents to a text file."
    )

    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to scrape for Parquet files.",
    )
    parser.add_argument(
        "-f",
        "--from_lang_code",
        type=str,
        required=True,
        help="from language code (e.g. en)",
    )
    parser.add_argument(
        "-t",
        "--to_lang_code",
        type=str,
        required=True,
        help="target langauge code (e.g. en)",
    )
    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default="input.txt",
        help="Path to the output text file where the contents should be saved.",
    )
    parser.add_argument(
        "--phonemize",
        action="store_true",
        help="emit to different files",
    )

    args = parser.parse_args()

    if args.url == None:
        args.url=f"https://huggingface.co/datasets/Helsinki-NLP/opus-100/tree/main/{args.from_lang_code}-{args.to_lang_code}"
    main(args.url, args.output_text_file, args.from_lang_code, args.to_lang_code, args.phonemize)
