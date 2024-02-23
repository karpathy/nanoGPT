#!/bin/bash

# Enhanced Bash script for processing and playing a MIDI file with cleanup on early exit

cleanup() {
    echo "Cleaning up temporary files..."
    rm -f "$temp_csv"
}

# Register cleanup function to be called on the EXIT signal
trap cleanup EXIT

# Check if an argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file="$1"
base_output_file="${input_file%.*}" # Removes extension from input file for naming

# Define temporary file for intermediate CSV
temp_csv=$(mktemp "${base_output_file}_XXXX.csv")

# Convert base to base and output as CSV
if ! python3 convert_base_to_base.py "$input_file" "$temp_csv" --input_base 12 --output_base 10; then
    echo "Error converting bases."
    exit 1
fi

# Convert CSV to MIDI
final_output="output.mid"
if ! python3 to_midi.py -i "$temp_csv" -o "$final_output"; then
    echo "Error converting CSV to MIDI."
    exit 1
fi

# Check for wildmidi installation and play the MIDI file
if ! command -v wildmidi &> /dev/null; then
    echo "wildmidi could not be found."
    echo "Install on Ubuntu with: sudo apt install wildmidi"
    exit 1
else
    wildmidi "$final_output"
fi
