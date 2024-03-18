import argparse
import os


def line_contains_forbidden_string(line, forbidden_strings):
    return any(forbidden_string in line for forbidden_string in forbidden_strings)


def interleave_files(
    file1_path,
    file2_path,
    start_with_file2,
    max_size_mb,
    output_prefix,
    output_folder,
    forbidden_strings,
):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    file_counter = 1
    output_file_path = os.path.join(output_folder, f"{output_prefix}_{file_counter}.txt")
    output_file = open(output_file_path, "w", encoding="utf-8")
    file_size_limit = max_size_mb * 1024 * 1024  # Convert MB to bytes

    with open(file1_path, "r", encoding="utf-8") as file1, open(file2_path, "r", encoding="utf-8") as file2:
        file1_lines = file1.readlines()
        file2_lines = file2.readlines()

        for line1, line2 in zip(file1_lines, file2_lines):
            lines = [line2, line1] if start_with_file2 else [line1, line2]
            skip_flag = False
            for line in lines:
                if line_contains_forbidden_string(line, forbidden_strings):
                    skip_flag = True # Skip writing the line if it contains any of the forbidden strings
                    break

            if skip_flag:
                continue
            for line in lines:
                output_file.write(line)

            output_file.write('\n')  # Write blank line after each pair

            if output_file.tell() > file_size_limit:
                output_file.close()
                file_counter += 1
                output_file_path = os.path.join(output_folder, f"{output_prefix}_{file_counter}.txt")
                output_file = open(output_file_path, "w", encoding="utf-8")

    output_file.close()

def main():
    parser = argparse.ArgumentParser(description="Interleave lines from two input text files.")
    parser.add_argument("-f1", "--file1", required=True, help="Path to the first input file.")
    parser.add_argument("-f2", "--file2", required=True, help="Path to the second input file.")
    parser.add_argument("-s2", "--start_with_file2", action="store_true", help="Start interleaving with the second file. If not set, starts with the first file.")
    parser.add_argument("-m", "--max_size_mb", type=float, required=True, help="Maximum size of each output file in megabytes (MB).")
    parser.add_argument("-o", "--output_prefix", required=True, help="Prefix for the output files.")
    parser.add_argument("--output_folder", default="interleaved_files", help="The folder to store output files.")
    parser.add_argument("--forbidden_strings", nargs='*', default=[], help="List of strings that, if present in a line, will cause the line to be skipped.")

    args = parser.parse_args()

    interleave_files(
        args.file1,
        args.file2,
        args.start_with_file2,
        args.max_size_mb,
        args.output_prefix,
        args.output_folder,
        args.forbidden_strings,
    )

if __name__ == "__main__":
    main()

