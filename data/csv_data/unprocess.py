import argparse
import re

def unshuffle_data(data, labels, convert_to_csv=False):
    """
    Unshuffles data based on the specified labels and prepends the part left of the first label to each line.

    :param data: List of strings containing shuffled data.
    :param labels: List of labels to order the data by.
    :return: List of strings with data unshuffled and prepended with the part left of the first label.
    """
    unshuffled_data = []

    for line in data:
        # Extracting the part before the first label
        prefix = re.match(r"[^" + "".join(labels) + "]*", line).group()

        # Extracting and sorting the parts with labels
        parts = re.findall(r"([" + "".join(labels) + "])(-?\d+\.\d+e[+-]\d+)", line)

        sorted_parts = sorted(
            parts, key=lambda x: labels.index(x[0]) if x[0] in labels else len(labels)
        )

        if convert_to_csv:
          # Prepending the prefix and forming the unshuffled line
          unshuffled_line = prefix + "".join(
              f",{value}" for label, value in sorted_parts
          )
        else:
          # Prepending the prefix and forming the unshuffled line
          unshuffled_line = prefix + "".join(
              f"{label}{value}" for label, value in sorted_parts
          )

        unshuffled_data.append(unshuffled_line)

    return unshuffled_data


def main():
    parser = argparse.ArgumentParser( description="Unshuffle data based on specified labels and prepend the part left of the first label.")
    parser.add_argument("-i", "--input_file", help="Input file containing the data to be unshuffled.")
    parser.add_argument("-o", "--output_file", nargs="?", help="Output file to write the unshuffled data. If not provided, prints to stdout.",)
    parser.add_argument("-l", "--labels", help='Labels used to order the data (e.g., "abc")')
    parser.add_argument("-c", "--convert_to_csv", action="store_true", help='converts file into csv')
    args = parser.parse_args()

    # Reading data from the input file
    with open(args.input_file, "r") as file:
        data = file.readlines()

    unshuffled_data = unshuffle_data(data, list(args.labels), convert_to_csv=args.convert_to_csv)

    if args.output_file:
        # Writing to the output file
        with open(args.output_file, "w") as file:
            for line in unshuffled_data:
                file.write(line + "\n")
    else:
        # Printing to stdout
        for line in unshuffled_data:
            print(line)


if __name__ == "__main__":
    main()
