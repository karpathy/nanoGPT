import os
import json
import requests
import argparse
from tqdm import tqdm
from bs4 import BeautifulSoup


def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Download incomplete.")
    else:
        print(f"Downloaded {filename}")


def emit_json_contents(
    json_path,
    output_text_file,
    include_keys,
    value_prefixes,
    required_key,
    skip_empty,
    exclude,
):
    print(f"Emitting contents from {json_path} to {output_text_file}")
    with open(json_path, "r") as f:
        data = json.load(f)
        if not isinstance(data, list):
            data = [data]

    print(f"Loaded {len(data)} items from {json_path}")

    skip_item = False
    excluded_pairs = {}
    if exclude:
        for pair in args.exclude:
            for i in range(0, len(pair), 2):
                key = pair[i]
                value = pair[i+1]
                excluded_pairs[key] = value

    with open(output_text_file, "w") as output_file:
        prev_item_written = False
        for item in data:
            if required_key and item.get(required_key, "") == "":
                print(f"Skipping item due to empty required key: {required_key}")
                continue
            for key, prefix in zip(include_keys, value_prefixes):
                key = key.strip("'\"")
                key = key.rstrip()
                if key in item:
                    if skip_empty and item[key] == "":
                        continue  # skip empty items if activated
                    for i_key, i_value in item.items():
                        if (i_key in excluded_pairs) and (i_value == excluded_pairs[i_key]):
                            skip_item=True
                        if skip_item:
                            break
                    if skip_item:
                        continue

                    if prev_item_written:
                        output_file.write("\n")

                    content_line = f"{prefix}{item[key]}"
                    output_file.write(content_line.strip())
                    prev_item_written = True


def find_file_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [
        "https://huggingface.co" + a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].endswith(".json?download=true") or a["href"].endswith(".jsonl?download=true")
    ]
    return links


def concatenate_json_files(json_paths, concatenated_json_path):
    with open(concatenated_json_path, "w") as output_file:
        output_file.write("[\n")
        first_item = True
        for json_path in json_paths:
            with open(json_path, "r") as input_file:
                for line in input_file:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Skip lines that are not valid JSON
                    if not first_item:
                        output_file.write(",\n")
                    json.dump(item, output_file)
                    first_item = False
        output_file.write("\n]")


def main(
    url,
    output_text_file,
    no_output_text,
    include_keys,
    value_prefixes,
    required_key,
    skip_empty,
    exclude,
    direct_json_input
):
    if direct_json_input is not None:
        emit_json_contents(
            direct_json_input,
            output_text_file,
            include_keys,
            value_prefixes,
            required_key,
            skip_empty,
            exclude,
            )
    else:
        file_links = find_file_links(url)
        download_dir = "./downloaded_jsons"
        json_dir = "./json_output"
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        json_paths = []
        for link in file_links:
            file_name = link.split("/")[-1].split("?")[0]
            file_path = os.path.join(download_dir, file_name)
            if not os.path.exists(file_path):
                download_file(link, file_path)
            json_paths.append(file_path)

        concatenated_json_path = os.path.join(json_dir, "concatenated.json")
        concatenate_json_files(json_paths, concatenated_json_path)

        if not no_output_text:
            emit_json_contents(
                concatenated_json_path,
                output_text_file,
                include_keys,
                value_prefixes,
                required_key,
                skip_empty,
                exclude,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape and convert JSON files from URL and save their contents to a text file."
    )
    parser.add_argument(
        "--url", type=str, help="URL to scrape for JSON files."
    )
    parser.add_argument(
        "-o",
        "--output_text_file",
        type=str,
        default="input.txt",
        help="Path to the output text file.",
    )
    parser.add_argument(
        "--no_output_text", action="store_true", help="Skip creation of output text."
    )
    parser.add_argument(
        "-i",
        "--include_keys",
        type=str,
        nargs="+",
        required=True,
        help="List of keys to include from the JSON contents.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        action="append",
        metavar=("KEY", "VALUE"),
        help="Specify key-value pairs to be excluded. Use the format: --exclude KEY VALUE [KEY VALUE ...]",
    )
    parser.add_argument(
        "-p",
        "--value_prefixes",
        type=str,
        nargs="+",
        default=[""],
        help="List of prefixes to be added to each value when emitting to the text file.",
    )
    parser.add_argument(
        "-r",
        "--required_key",
        type=str,
        default=None,
        help="Only emit items that have this key (optional).",
    )
    parser.add_argument(
        "-s",
        "--skip_empty",
        action="store_true",
        help="Skip any item which is an empty string.",
    )
    parser.add_argument(
        "-j",
        "--direct_json_input",
        type=str,
        default=None,
        help="skip download and process with manual json or jsonl input",
    )

    args = parser.parse_args()
    main(
        args.url,
        args.output_text_file,
        args.no_output_text,
        args.include_keys,
        args.value_prefixes,
        args.required_key,
        args.skip_empty,
        args.exclude,
        args.direct_json_input,
    )
