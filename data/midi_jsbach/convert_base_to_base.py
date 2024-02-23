import argparse
import csv

def process_csv(input_file, output_file, input_base, output_base):
    with open(input_file, 'r', newline='') as csvfile_in, open(output_file, 'w', newline='') as csvfile_out:
        reader = csv.reader(csvfile_in)
        writer = csv.writer(csvfile_out)

        for row in reader:
            converted_row = [convert_from_base_to_base(num, input_base, output_base) for num in row]
            writer.writerow(converted_row)

def convert_from_base_to_base(num_str, input_base, output_base):
    """
    Converts a number from an input base to an output base.
    """
    if num_str == "~":
        return "-1"
    # Convert from input base to decimal
    num_decimal = int(num_str, input_base)
    # Convert from decimal to output base
    return convert_to_base(num_decimal, output_base)

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
    parser = argparse.ArgumentParser(description='Convert numbers from one base to another in a CSV file.')
    parser.add_argument('input_file', type=str, help='The path to the input CSV file')
    parser.add_argument('output_file', type=str, help='The path to the output CSV file')
    parser.add_argument('--input_base', type=int, required=True, help='The base of numbers in the input CSV file')
    parser.add_argument('--output_base', type=int, required=True, help='The base to convert each number to in the output CSV file')

    args = parser.parse_args()

    process_csv(args.input_file, args.output_file, args.input_base, args.output_base)

