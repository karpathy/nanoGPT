import argparse
import json

def parseargs():
    parser = argparse.ArgumentParser(description='Extract questions and answers from a JSONL file.')
    parser.add_argument('input_file', type=str, help='path to input jsonl file')
    parser.add_argument('output_file', type=str, help='Path to output txt file')

    return parser.parse_args()

# Function to extract and format data from each JSON object in the .jsonl file
def extract_and_format_data(file_path):
    formatted_data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data_dict = json.loads(line)
            question = data_dict.get("question", "")
            expected_answer = data_dict.get("expected_answer", "")
            formatted_data = f"question: {question}\nanswer: {expected_answer}\n"
            formatted_data_list.append(formatted_data)
    return formatted_data_list

# Write the extracted and formatted data to a text file
def write_to_file(formatted_data_list, output_file):
    with open(output_file, "w") as file:
        for formatted_data in formatted_data_list:
            file.write(formatted_data + "\n")  # Add an extra newline for separation between entries

# Main script execution
if __name__ == "__main__":
    args = parseargs()
    formatted_data_list = extract_and_format_data(args.input_file)
    write_to_file(formatted_data_list, args.output_file)
    print(f"Data has been extracted and saved to '{args.output_file}'.")

