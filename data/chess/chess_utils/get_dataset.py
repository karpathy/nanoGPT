import os
import argparse
import requests
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn

def download_games(url, output_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if the file is already downloaded and decompressed
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping download.")
        return

    # Perform the actual download
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    console = Console()
    console.log("Starting download...")

    with open(output_path, 'wb') as file:
        with Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=None),
            TransferSpeedColumn(),  # Displays the download speed
            TimeRemainingColumn(),  # Estimates the time remaining for the download
            console=console,
            transient=True
        ) as progress:
            download_task = progress.add_task("Downloading", total=total_size, filename=os.path.basename(url))
            for chunk in response.iter_content(chunk_size=16384):
                # Directly write the downloaded chunk to the file
                file.write(chunk)
                progress.update(download_task, advance=len(chunk))

    console.log("Download completed.")

def setup_argparse():
    parser = argparse.ArgumentParser(description='Download and decompress Lichess dataset')
    default_url = f'https://database.lichess.org/standard/lichess_db_standard_rated_2015-01.pgn.zst'
    parser.add_argument('--url', type=str, default=default_url, help='URL of the file to download')
    return parser.parse_args()

def main():
    args = setup_argparse()
    dataset_dir = 'datasets'
    zst_file_name = 'lichess_games.zst'
    txt_output_path = os.path.join(dataset_dir, zst_file_name)

    download_games(args.url, txt_output_path)
    print("Conversion to JSON completed.")

if __name__ == '__main__':
    main()

