#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_PARQUET_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.

# Modify for the url
url="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/main/data"

# Run the Python script with the specified arguments
# Note: the $'\n' syntax allows for special characters like \n
python3 ./utils/get_parquet_dataset.py \
  --url "${url}" \
  --include_keys "question" "options" "answer" \
  --value_prefix $'#U:\n' "options: " $'#B:\n' \
  --skip_empty

# U for user 'B' for bot
#
#
#!/bin/bash

# Create a temporary file
temp_file=$(mktemp)

# Function to clean up the temporary file
cleanup() {
    rm -f "$temp_file"
}
trap cleanup EXIT

# Check if the input file exists
if [ ! -f "input.txt" ]; then
    echo "File input.txt not found!"
    exit 1
fi

# Get the total number of lines in the input file
total_lines=$(wc -l < input.txt)

# Read the input file line by line with progress bar
pv -l -s $total_lines input.txt | while IFS= read -r line
do
    # Check if the line starts with "options:"
    if [[ $line == options:* ]]; then
        # Extract the options part and remove the surrounding characters
        options=$(echo "$line" | sed -E 's/options: \[//; s/\]//; s/'\''//g')

        # Split the options by comma and add alphabetical prefixes
        IFS=',' read -ra ADDR <<< "$options"
        prefix=A
        for i in "${ADDR[@]}"; do
            # Remove leading and trailing spaces
            option=$(echo "$i" | sed 's/^ *//; s/ *$//')
            echo "$prefix) $option" >> "$temp_file"
            prefix=$(echo "$prefix" | tr "0-9A-Z" "1-9A-Z_")
        done
    else
        # Print the original line if it doesn't start with "options:"
        echo "$line" >> "$temp_file"
    fi
done

# Overwrite the original input file with the temporary file
mv "$temp_file" "input.txt"

