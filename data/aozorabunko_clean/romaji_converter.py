import argparse
from tqdm import tqdm
import cutlet

def parse_args():
    parser = argparse.ArgumentParser(description="Convert sentences in a file to romaji.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    parser.add_argument("--system", choices=["hepburn", "kunrei", "nippon", "nihon"], default="hepburn", help="Romanization system to use (default: hepburn)")
    parser.add_argument("--use-foreign-spelling", action="store_true", help="Use foreign spelling when available (default: False)")
    return parser.parse_args()

def main():
    args = parse_args()
    katsu = cutlet.Cutlet(args.system)
    katsu.use_foreign_spelling = args.use_foreign_spelling

    with open(args.input_file, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()  # Read all lines at once to count them for progress bar

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        # Process each line with a progress bar
        for line in tqdm(lines, desc="Converting", unit="lines"):
            words = line.rstrip().split()  # Split the line into words to maintain spaces
            romaji_words = [katsu.romaji(word) for word in words]  # Convert each word to romaji
            output_file.write(' '.join(romaji_words) + "\n")  # Re-join words with a space

if __name__ == "__main__":
    main()

