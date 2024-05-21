import os
import zstandard as zstd
from rich.console import Console

def decompress_file(input_zst_path, output_txt_path):
    console = Console()

    # Check if the input file exists
    if not os.path.exists(input_zst_path):
        console.log(f"Input file {input_zst_path} does not exist. Exiting.")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    # Perform decompression using a streaming approach
    with open(input_zst_path, 'rb') as compressed_file, \
         open(output_txt_path, 'wb') as decompressed_file, \
         console.status("[bold green]Decompressing...", spinner="dots"):
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(compressed_file)
        while True:
            chunk = reader.read(16384)  # Read chunks of 16 KB
            if not chunk:
                break
            decompressed_file.write(chunk)
    console.log("Decompression completed successfully.")

if __name__ == '__main__':
    dataset_dir = 'datasets'
    zst_file_name = 'lichess_games.zst'
    txt_file_name = 'lichess_games.txt'

    input_zst_path = os.path.join(dataset_dir, zst_file_name)
    output_txt_path = os.path.join(dataset_dir, txt_file_name)

    decompress_file(input_zst_path, output_txt_path)

