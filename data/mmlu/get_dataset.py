import argparse
import json
import os
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import pandas as pd

def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        raise Exception("Error: Failed to download the file completely.")
    print(f"Downloaded {filename}")

def convert_to_json(parquet_path, json_path):
    if not os.path.exists(json_path):
        df = pd.read_parquet(parquet_path)
        df.to_json(json_path, orient="records")
        print(f"Converted {parquet_path} to JSON at {json_path}")
    else:
        print(f"{json_path} already exists, skipping conversion.")

def emit_json_contents(json_path, output_text_file):
    with open(json_path, "r") as f:
        data = json.load(f)
    with open(output_text_file, "a") as f:
        for item in data:
            f.write(f"Question: {item['question']}\n")
            for idx, choice in enumerate(item['choices']):
                f.write(f"Choice {idx + 1}: {choice}\n")
            correct_answer = item['choices'][item['answer']]
            f.write(f"Correct Answer: Choice {item['answer'] + 1}: {correct_answer}\n\n")

def find_parquet_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    base_url = "https://huggingface.co"
    links = [base_url + a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.parquet?download=true')]
    return links

def main(base_url, subdirectories, output_text_file):
    download_dir = "./downloaded_parquets"
    json_dir = "./json_output"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    open(output_text_file, "w").close()

    for subdir in subdirectories:
        url = f"{base_url}/{subdir}/"
        print(f"Downloading .parquet files for {subdir}...")
        links = find_parquet_links(url)
        for link in links:
            print(f"Found file: {link}")
            file_name = f"{subdir}_{os.path.basename(link.split('?')[0])}"
            parquet_path = os.path.join(download_dir, file_name)
            json_path = os.path.join(json_dir, file_name.replace('.parquet', '.json'))
            if not os.path.isfile(parquet_path):
                download_file(link, parquet_path)
            if not os.path.exists(json_path):
                convert_to_json(parquet_path, json_path)
            emit_json_contents(json_path, output_text_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape specific subdirectories for Parquet files.")
    parser.add_argument("--output_text_file", type=str, default="input.txt", help="Path to the output text file.")
    args = parser.parse_args()
    base_url = "https://huggingface.co/datasets/cais/mmlu/tree/main"
    subdirectories = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics",
        "computer_security", "virology"
    ]
    main(base_url, subdirectories, args.output_text_file)

