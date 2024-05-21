import re
import json
import argparse
import os

def extract_movesets(json_file, output_file, remove_numbers=False):
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as file:
        for entry in data:
            if 'Moveset' in entry:
                moveset = entry['Moveset']
                if not any(char in moveset for char in ['%', '{', '}', '[', ']', '?']):
                    cleaned_moveset = re.sub(r'\s(1-0|0-1|1/2-1/2)$', '', moveset)
                    if remove_numbers:
                        cleaned_moveset = re.sub(r'[0-9]+\. ', '', cleaned_moveset)
                    file.write(cleaned_moveset + '\n')

def main():
    parser = argparse.ArgumentParser(description="Extract and clean movesets from JSON file.")
    parser.add_argument('-i', '--json_file', default='filtered_json/filtered_games.json', type=str, help='Path to the JSON file')
    parser.add_argument('-o', '--output_file', default='movesets_txt/moveset.txt', type=str, help='Path to the output text file where cleaned movesets will be saved')
    parser.add_argument('-r', '--remove_numbers', default=True,
                             action=argparse.BooleanOptionalAction,
                             help="remove numbers")

    args = parser.parse_args()
    extract_movesets(args.json_file, args.output_file, args.remove_numbers)

if __name__ == '__main__':
    main()
