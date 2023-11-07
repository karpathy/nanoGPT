import csv
import argparse

def combine_csv_columns(file_path1, file_path2, output_file_path):
    with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2, open(output_file_path, 'w', newline='') as outfile:
        reader1 = csv.reader(file1)
        reader2 = csv.reader(file2)

        for row1, row2 in zip(reader1, reader2):
            # Combine the rows from both CSVs
            combined_row = row1 + row2

            # Join the row with no delimiter and write it to the file
            outfile.write(''.join(combined_row))

            # Write a newline character after each row
            outfile.write('\n')

def main(args):
    combine_csv_columns(args.file1, args.file2, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine columns of two CSV files.')
    parser.add_argument('file1', type=str, help='Path to the first input CSV file.')
    parser.add_argument('file2', type=str, help='Path to the second input CSV file.')
    parser.add_argument('output', type=str, help='Path to the output CSV file.')

    args = parser.parse_args()
    main(args)

