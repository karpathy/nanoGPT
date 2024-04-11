#!/bin/bash

# Function to display help menu
show_help() {
    echo "Usage: $0 [options] input_file output_file"
    echo
    echo "Options:"
    echo "  -h             Display this help message and exit."
    echo "  -n <number>    Set the number of cores to use. Defaults to all cores."
    echo "  -l <lang_code> Espeak language code (e.g. en, fr, id, ...) "
    echo "  -o             Option to remove any newlines per espeak output"
    echo
    echo "Example:"
    echo "  $0 -n 4 -l fr -o input.txt output.txt"
    echo
    echo "This script reads from an input file, processes each line, and writes the french phoneme output to an output file."
}

# Default number of cores to use
num_cores=""
no_newlines=""
language="en"

# Parse command-line options
while getopts ":hn:l:o" opt; do
    case ${opt} in
        h )
            show_help
            exit 0
            ;;
        n )
            num_cores="-j ${OPTARG}"
            ;;
        l )
            language="${OPTARG}"
            ;;
        o )
            no_newlines="true"
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            exit 1
            ;;
        : )
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            exit 1
            ;;
    esac
done

# Remove the options processed above
shift $((OPTIND -1))

# Check if the correct number of arguments is given after option processing
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 [options] input_file output_file"
    exit 1
fi

# Assign input and output file from arguments
input_file=$1
output_file=$2

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file does not exist"
    exit 1
fi

# Function to process a line
process_line() {
    line="$1"
    local no_newlines="${2}"
    local language="${3}"

    if [[ "${no_newlines}" = "true" ]]; then
      echo "$line" | espeak-ng -q -x -v "${language}" | tr -d '\n'
      echo ""
    else
      echo "$line" | espeak-ng -q -x -v "${language}"
    fi
}

# Export the function to be used by parallel
export -f process_line

# Use GNU Parallel to process the lines using specified number of cores
cat "$input_file" | parallel $num_cores -k process_line {} "$no_newlines" "$language" >> "$output_file"

