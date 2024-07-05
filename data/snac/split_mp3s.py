import os
import subprocess
import argparse

def get_file_size(path):
    return os.path.getsize(path)

def split_mp3(input_file, max_size_mb, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Calculate the maximum size in bytes
    max_size_bytes = max_size_mb * 1024 * 1024

    # Use ffmpeg to get the duration of the input file
    result = subprocess.run(
        ['ffmpeg', '-i', input_file, '-f', 'null', '-'],
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    duration = None
    for line in result.stderr.split('\n'):
        if 'Duration' in line:
            duration_str = line.split()[1].strip(',')
            h, m, s = map(float, duration_str.split(':'))
            duration = h * 3600 + m * 60 + s
            break

    if duration is None:
        print(f"Could not determine duration of {input_file}")
        return

    # Calculate approximate duration per chunk
    input_size = get_file_size(input_file)
    chunk_duration = (max_size_bytes / input_size) * duration

    # Split the file using ffmpeg
    cmd = [
        'ffmpeg', '-i', input_file, '-f', 'segment', '-segment_time', str(chunk_duration), '-c', 'copy',
        os.path.join(output_dir, f"{base_name}_part%03d.mp3")
    ]
    subprocess.run(cmd, check=True)
    print(f"Split {input_file} into chunks of max {max_size_mb} MB in {output_dir}")

def split_directory(input_dir, max_size_mb, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.mp3'):
            input_file = os.path.join(input_dir, filename)
            split_mp3(input_file, max_size_mb, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Split MP3 files in a directory into chunks of a specified maximum size.")
    parser.add_argument('input_dir', help="Directory containing MP3 files to split")
    parser.add_argument('--max_size_mb', type=int, default=5, help="Maximum size of each chunk in megabytes (default: 5 MB)")
    parser.add_argument('--output_dir', help="Directory to save split files (default: split_<input_dir>)")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output_dir or f"split_{os.path.basename(input_dir)}"

    split_directory(input_dir, args.max_size_mb, output_dir)

if __name__ == "__main__":
    main()

