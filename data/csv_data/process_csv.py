import csv
import argparse
import random

def create_letter_mapping(exclude: list) -> dict:
    # Create a mapping of indices to letters, skipping excluded letters.
    allowed_letters = [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) not in exclude]
    return {i: letter for i, letter in enumerate(allowed_letters)}

def process_csv(input_file: str, output_file: str, shuffle: bool, exclude: list) -> None:
    # Create the letter mapping
    letter_mapping = create_letter_mapping(exclude)

    with open(input_file, mode="r") as csv_file, open(output_file, mode="w") as txt_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            # Use the letter mapping to assign letters to values
            letter_value_pairs = [
                f"{letter_mapping[i]}{val}" for i, val in enumerate(row) if i in letter_mapping
            ]

            if shuffle:
                random.shuffle(letter_value_pairs)

            # Join the letter-value pairs with no spaces and write to the output file.
            txt_file.write("".join(letter_value_pairs) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a time-series CSV and convert it to a custom text format while excluding certain letters."
    )
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_file", type=str, help="Path to the output text file.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the order of letter and value pairs.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="A list of letters to exclude from the letter labeling.",
    )

    args = parser.parse_args()
    process_csv(args.input_file, args.output_file, args.shuffle, args.exclude)

