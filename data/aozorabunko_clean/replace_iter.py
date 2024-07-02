import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Replace iteration marks in a file.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    return parser.parse_args()

def replace_iteration_marks(line):
    result = []
    last_char = ''
    for i, char in enumerate(line):
        if char == "ゝ":
            if last_char:
                result.append(last_char)
        else:
            result.append(char)
            last_char = char  # Update last_char only if the current char is not an iteration mark
    return ''.join(result)

def main():
    args = parse_args()

    with open(args.input_file, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()  # Read all lines at once to count them for progress bar

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        # Process each line with a progress bar
        for line in tqdm(lines, desc="Processing", unit="lines"):
            modified_line = replace_iteration_marks(line)
            output_file.write(modified_line)

if __name__ == "__main__":
    main()

