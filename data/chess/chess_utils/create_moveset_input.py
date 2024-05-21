import re
import argparse

def preprocess_movesets(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    preprocessed_lines = []
    for line in lines:
        # Replace " numeral." with underscore
        preprocessed_line = re.sub(r' (\d+\.)', r'_\1', line)

        # Replace ". " with lowercase omega
        preprocessed_line = re.sub(r'\. ', 'ω', preprocessed_line)

        # Replace remaining with beta
        preprocessed_line = re.sub(r' ', r'β', preprocessed_line)

        preprocessed_lines.append(preprocessed_line)

    with open(output_file, 'w') as file:
        file.write(''.join(preprocessed_lines))

def main():
    parser = argparse.ArgumentParser(description="Preprocess movesets.")
    parser.add_argument('-i', '--input_file', default="movesets_txt/moveset.txt", type=str, help='Path to the input text file containing movesets')
    parser.add_argument('-o', '--output_file', default="input.txt", type=str, help='Path to the output text file where preprocessed movesets will be saved')
    args = parser.parse_args()
    preprocess_movesets(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
