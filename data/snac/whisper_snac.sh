#!/bin/bash

# Set strict error handling
set -xeuo pipefail

# Install the dependencies needed
sudo apt install -y ffmpeg

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "This script will perform whisper, transfer the results to snac directory, and run the corresponding program to get snac-word json files."

whisper_cpp_dir="${script_dir}/../template/whisper.cpp"

# Check if whisper.cpp exists, prompt to initialize submodule
if [ ! -d "$whisper_cpp_dir" ]; then
    echo "Please run the whisper_install.sh first to install whisper.cpp."
    exit 1
fi

# Check if whisper.cpp is compiled
function check_file_exists {
    if [[ ! -f "$1" ]]; then
        echo "whisper.cpp is not compiled, please run the whisper_install.sh."
        exit 1
    fi
}

check_file_exists ${whisper_cpp_dir}/main

# Download the audio dataset for whisper.cpp to process
url="https://huggingface.co/datasets/eastwind/tiny-sherlock-audio"
out_dir="input_mp3s"

if [[ ! -d "${out_dir}" ]]; then
    mkdir -p "${out_dir}"
fi

for i in $(seq -f "%02g" 1 24); do
    wget -nc -O "${out_dir}/tiny_sherlock_audio_${i}.mp3" "${url}/resolve/main/adventuresholmes_${i}_doyle_64kb.mp3?download=true" || true
done

# python3 split_mp3s.py "${out_dir}" --max_size_mb 5

# Using the whisper.cpp to get the section of words to the audio and save the output in ~/snac
whisper_json_dir="whisper_json_dir"

if [[ ! -d "${whisper_json_dir}" ]]; then
    mkdir -p "${whisper_json_dir}"
fi

result_dir="output_jsons"

if [[ ! -d "${result_dir}" ]]; then
    mkdir -p "${result_dir}"
fi

pushd "${whisper_json_dir}"

# Loop through all .mp3 files in the current directory
for mp3_file in ../split_input_mp3s/*.mp3; do
    echo "$mp3_file"

    # Get the base name of the file without the extension
    base_name="$(basename ${mp3_file%.mp3})"
    wav_file="${mp3_file%.mp3}.wav"
    echo "$base_name"
    echo "$wav_file"

    # Define the output .wav file name
    # wav_file="${base_name}.wav"

    # Convert .mp3 to .wav using ffmpeg
    # TODO: is this the right hz?
    ffmpeg -i "$mp3_file" -ar 16000 -y "$wav_file"

    # echo "Converted ${mp3_file} to ${wav_file}"

    # out_path="${output_dir}/${base_name}"

    # Run the whisper.cpp
    "${script_dir}/../template/whisper.cpp/main" -m "${script_dir}/../template/whisper.cpp/models/ggml-base.en.bin" -f "${wav_file}" -ml 1 -oj -of "${base_name}"

    # Run the sample_whisper_snac.py to get the json output
    result_name="${base_name}.json"
    result_path="../${result_dir}/${result_name}"

    python3 ../sample_whisper_snac.py "${mp3_file}" "${base_name}.json" "${result_path}"

    echo "Finished running ${wav_file }and saved results to ${result_path}"
done

popd

echo "All conversions are done."
