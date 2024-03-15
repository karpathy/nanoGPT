import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from taibun import Converter

def convert_to_zhuyin(text):
    """
    Converts a given text to Zhuyin using the taibun Converter.
    """
    converter = Converter(system='Zhuyin')
    return converter.get(text)

def process_file(input_file, output_file):
    """
    Reads an input file, converts each line to Zhuyin, and writes the output to an output file.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile, \
         ThreadPoolExecutor() as executor:

        # Submit conversion tasks for each line in the input file
        future_to_line = {executor.submit(convert_to_zhuyin, line.strip()): line for line in infile}

        for future in as_completed(future_to_line):
            zhuyin_line = future.result()
            outfile.write(zhuyin_line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert text file to Zhuyin.')
    parser.add_argument('-i', '--input', default='input.txt', help='Input file name')
    parser.add_argument('-o', '--output', default='zhuyin.txt', help='Output file name')

    args = parser.parse_args()

    process_file(args.input, args.output)
    print(f"Conversion completed. Output written to {args.output}")

