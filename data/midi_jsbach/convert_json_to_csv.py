import json
import csv
import argparse

def write_chords_to_csv(data, key, filename):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)

        for song in data[key]:
            for chord in song:
                writer.writerow(chord)

def main(input_json, output_csv):
    # Load JSON data from file
    with open(input_json, 'r') as file:
        data = json.load(file)

    # Append chords to output_csv file
    for key in data.keys():
        write_chords_to_csv(data, key, output_csv)

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Convert MIDI JSON data to CSV.')
    parser.add_argument('-i', '--input', type=str, default="midi.json", help='Input JSON file path')
    parser.add_argument('-o', '--output', type=str, default="midi.csv", help='Output CSV file path')

    # Parse arguments
    args = parser.parse_args()

    # convert json input to csv output
    main(args.input, args.output)


