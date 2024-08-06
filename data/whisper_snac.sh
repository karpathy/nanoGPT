#!/bin/bash

# Set strict error handling
set -euo pipefail

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "This script will perform whisper, transfer the results to snac directory, and run the corresponding program to get snac-word json files."

# Check if whisper.cpp exists, prompt to initialize submodule
if [ ! -d "$script_dir/template/whisper.cpp" ]; then
    echo "Please run the whisper_install.sh first to install whisper.cpp."
    exit 1
fi

pushd "$script_dir/template/whisper.cpp"

# Download the audio dataset for whisper.cpp to process
url="https://huggingface.co/datasets/eastwind/tiny-sherlock-audio"
out_dir="input_mp3s"

if [[ ! -d "${out_dir}" ]]; then
    mkdir -p "${out_dir}"
fi

for i in $(seq -f "%02g" 1 24); do
    wget -nc -O "${out_dir}/tiny_sherlock_audio_${i}.mp3" "${url}/resolve/main/adventuresholmes_${i}_doyle_64kb.mp3?download=true"
done

# Save the copy of this dataset to snac directory for later use
cp -r "$script_dir/template/whisper.cpp/${out_dir}" "$script_dir/snac"

# Install the dependencies needed
sudo apt install ffmpeg

popd

pushd "$script_dir/template/whisper.cpp/input_mp3s"

# Loop through all .mp3 files in the current directory
for mp3_file in *.mp3; do
    # Check if any .mp3 files are present
    if [[ ! -e "$mp3_file" ]]; then
        echo "No .mp3 files found in the current directory."
        exit 1
    fi
    
    # Get the base name of the file without the extension
    base_name="${mp3_file%.mp3}"
    
    # Define the output .wav file name
    wav_file="$base_name.wav"
    
    # Convert .mp3 to .wav using ffmpeg
    ffmpeg -i "$mp3_file" -ar 16000 -y "$wav_file"
    
    echo "Converted $mp3_file to $wav_file"
done

echo "All conversions are done."

popd

# Using the whisper.cpp to get the section of words to the audio and save the output in ~/snac
pushd "$script_dir/template/whisper.cpp"
output_dir="input_words"

if [[ ! -d "${output_dir}" ]]; then
    mkdir -p "${output_dir}"
fi

# Loop through all the .wav files in the input directory
for input_audio in ${out_dir}/*.wav; do
    # Extract the base name of the file to use it for the output file
    out_name=$(basename "${input_audio%.wav}")
    out_path="${output_dir}/${out_name}"

    # Run the whisper.cpp
    ./main -m ./models/ggml-base.en.bin -f "${input_audio}" -ml 1 -oj -of "${out_path}"
done

echo "Finished running whisper.cpp"

popd

pushd "$script_dir/snac"

# Move the ouput files in whisper.cpp to this directory
mv "$script_dir/template/whisper.cpp/${output_dir}" "$script_dir/snac"

popd
