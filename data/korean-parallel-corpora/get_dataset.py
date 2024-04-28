import pandas as pd
import requests
import os
import argparse
import json
from tqdm import tqdm
from jamo import h2j, j2hcj, is_hangul_char

def download_file(url, filename):
    """
    Download a file from a given URL with a progress bar, only if it is not already present.
    """
    if os.path.exists(filename):
        print(f"{filename} already downloaded.")
        return
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the download was successful.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading file")
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Failed to download the file completely.")
    else:
        print(f"Downloaded {filename}")

def format_hangul_text(text):
    """Format the text so that each Hangul character is surrounded by exactly one space."""
    formatted_text = []
    for char in text:
        if is_hangul_char(char):
            formatted_text.append(' ' + char + ' ')
        else:
            formatted_text.append(char)
    return ''.join(formatted_text)

def korean_to_phonetic(text):
    """Converts Korean text to its phonetic representation."""
    text = text.replace(' ', '▁')
    text = format_hangul_text(text)
    decomposed_text = h2j(text)
    phonetic_text = j2hcj(decomposed_text)
    return phonetic_text

def process_csv(csv_path, json_path, txt_path, order, prefixes):
    """
    Process CSV file and save data as JSON and text with progress bars.
    """
    df = pd.read_csv(csv_path, keep_default_na=False)
    data = []
    progress_bar = tqdm(total=len(df), desc="Processing records", unit="records")
    with open(txt_path, 'w') as txt_file:
        for _, row in df.iterrows():
            entries = {
                'ko': '\n' + prefixes['ko'] + row['ko'],
                'en': '\n' + prefixes['en'] + row['en'],
                'ph': '\n' + prefixes['ph'] + korean_to_phonetic(row['ko'])
            }
            line_content = [entries[item] for item in order if item in entries]
            txt_file.write(' '.join(line_content) + "\n")
            data.append({'ko': row['ko'], 'en': row['en'], 'ph': entries['ph']})
            progress_bar.update(1)
    progress_bar.close()
    with open(json_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Converted {csv_path} to JSON at {json_path} and to text at {txt_path}")

def main(url, json_path, txt_path, order, ko_prefix, en_prefix, ph_prefix):
    download_dir = "./downloaded_files"
    os.makedirs(download_dir, exist_ok=True)
    file_name = url.split("/")[-1].split("?")[0]
    csv_path = os.path.join(download_dir, file_name)
    download_file(url, csv_path)
    process_csv(csv_path, json_path, txt_path, order.split(','), {'ko': ko_prefix, 'en': en_prefix, 'ph': ph_prefix})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CSV file to JSON and text file with phonetic representation of Korean text."
    )
    parser.add_argument("--url", type=str, default="https://huggingface.co/datasets/Moo/korean-parallel-corpora/resolve/main/train.csv?download=true", help="URL to download the CSV file from.")
    parser.add_argument("-o", "--output_json_file", type=str, default="data.json", help="Path to the output JSON file.")
    parser.add_argument("-t", "--output_txt_file", type=str, default="input.txt", help="Path to the output text file.")
    parser.add_argument("--order", type=str, default="ko,en,ph", help="Comma-separated order of fields to write to the text file: ko, en, ph.")
    parser.add_argument("--ko_prefix", type=str, default="ko:", help="Prefix for Korean text.")
    parser.add_argument("--en_prefix", type=str, default="en:", help="Prefix for English text.")
    parser.add_argument("--ph_prefix", type=str, default="ph:", help="Prefix for Phonetic text.")
    args = parser.parse_args()
    main(args.url, args.output_json_file, args.output_txt_file, args.order, args.ko_prefix, args.en_prefix, args.ph_prefix)

