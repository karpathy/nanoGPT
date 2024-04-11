import argparse
import json
import os
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import requests

def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the download was successful.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    if total_size and progress_bar.n != total_size:
        raise Exception("Error: Failed to download the file completely.")
    print(f"Downloaded {filename}")

def convert_to_json(parquet_path, json_path):
    if not os.path.exists(json_path):
        df = pd.read_parquet(parquet_path)
        df.to_json(json_path, orient="records")
        print(f"Converted {parquet_path} to JSON at {json_path}")
    print(f"{json_path} already exists, skipping conversion.")

def emit_json_contents(json_path, output_text_file):
    with open(json_path, "r") as f:
        data = json.load(f)

    with open(output_text_file, "a") as f:
        for item in data:
            content_line = f"{item['prompt']}"
            f.write(content_line.strip())
            f.write("\n")  # Separator between prompts and texts
            content_line = f"{item['text']}"
            f.write(content_line.strip())
            f.write("\n\n")  # Separator between entries

def find_parquet_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    base_url = "https://huggingface.co"
    links = [base_url + a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.parquet?download=true')]
    return links

def main(url, output_text_file):

    download_dir = "./downloaded_parquets"
    json_dir = "./json_output"

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # Ensure the output text file is empty before starting
    open(output_text_file, "w").close()

    for link in find_parquet_links(url):
        file_name = os.path.basename(link.split("?")[0])  # Extract filename
        parquet_path = os.path.join(download_dir, file_name)
        json_path = os.path.join(json_dir, file_name.replace('.parquet', '.json'))

        if not os.path.isfile(parquet_path):
            download_file(link, parquet_path)  # Download if not present

        convert_to_json(parquet_path, json_path)  # Convert to JSON

        emit_json_contents(json_path, output_text_file)  # Emit contents


if __name__ == "__main__":
    description = "Scrape and convert Parquet files from URL to JSON and save its contents to a text file."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--url", type=str, required=True, help="URL to scrape for Parquet files.")
    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default="input.txt",
        help="Path to the output text file.",
    )
    args = parser.parse_args()

    main(args.url, args.output_text_file)
