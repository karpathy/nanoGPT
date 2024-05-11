import argparse
import csv

def process_csv(input_file, output_file, base):
    with open(input_file, 'r', newline='') as csvfile_in, open(output_file, 'w', newline='') as csvfile_out:
        reader = csv.reader(csvfile_in)
        writer = csv.writer(csvfile_out)

        for row in reader:
            new_base_row = [convert_to_base(int(num), base) for num in row]
            writer.writerow(new_base_row)

def convert_to_base(num, base):
    """
    Converts a number to a specified base and returns it as a string.
    Supports bases from 2 to 36.
    """
    if num == -1:
        return "~"
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36.")
    
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if num == 0:
        return "0"
    result = ""
    while num > 0:
        result = digits[num % base] + result
        num //= base
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a CSV file by applying modulo operation to each number and converting it to a specified base.')
    parser.add_argument('input_file', type=str, help='The path to the input CSV file')
    parser.add_argument('output_file', type=str, help='The path to the output CSV file')
    parser.add_argument('--base', type=int, required=True, help='The base to convert each number to after applying the modulo operation')

    args = parser.parse_args()

    process_csv(args.input_file, args.output_file, args.base)

