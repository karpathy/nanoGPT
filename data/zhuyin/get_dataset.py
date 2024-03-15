import requests
from tqdm import tqdm
import argparse
import os

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

def main(language_code):
    base_url = "https://data.statmt.org/cc-100/"
    file_extension = ".txt.xz"
    file_url = f"{base_url}{language_code}{file_extension}"
    output_filename = "input.txt"

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Download the file
    download_file(file_url, output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a language-specific file."
    )

    parser.add_argument(
        "-c",
        "--code",
        type=str,
        default="zh-Hans",
        help="Language code for the file to be downloaded.",
    )

    args = parser.parse_args()
    main(args.code)

