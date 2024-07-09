import argparse

def process_simple(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            token = line.strip()
            if token == '4097':
                outfile.write('\n4097\n')
            else:
                outfile.write(f'{token} ')

def process_structured(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        buffer = []
        for line in infile:
            token = line.strip()
            if token == '4097':
                if buffer:
                    if len(buffer) == 7:
                        outfile.write('4097\n')
                        outfile.write(f'{buffer[0]}\n')
                        outfile.write(f'{buffer[1]} {buffer[2]}\n')
                        outfile.write(f'{buffer[3]} {buffer[4]} {buffer[5]} {buffer[6]}\n\n')
                    buffer = []
                else:
                    outfile.write('4097\n')
            else:
                buffer.append(token)
                if len(buffer) > 7:
                    buffer = []

def process_clean(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        buffer = []
        in_section = False
        for line in infile:
            token = line.strip()
            if token == '4097':
                if in_section:
                    if len(buffer) == 7:
                        outfile.write('4097\n')
                        outfile.write('\n'.join(buffer) + '\n')
                    buffer = []
                in_section = True
            elif in_section:
                buffer.append(token)
                if len(buffer) > 7:
                    buffer = []
                    in_section = False

        # Handle the last section if it's valid
        if in_section and len(buffer) == 7:
            outfile.write('4097\n')
            outfile.write('\n'.join(buffer) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Process a text file of tokens.")
    parser.add_argument('mode', choices=['simple', 'structured', 'clean'], help="Processing mode")
    parser.add_argument('input', help="Input file path")
    parser.add_argument('output', help="Output file path")
    args = parser.parse_args()

    if args.mode == 'simple':
        process_simple(args.input, args.output)
    elif args.mode == 'structured':
        process_structured(args.input, args.output)
    elif args.mode == 'clean':
        process_clean(args.input, args.output)

if __name__ == "__main__":
    main()
