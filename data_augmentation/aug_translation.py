import json
import argparse
import argostranslate.package
import argostranslate.translate

def parse_args():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input JSON file")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("-j", "--json", action="store_true", help="Create json")
    parser.add_argument("-t", "--translate", action="store_true", help="Enable translation")
    parser.add_argument('-l', "--to_code", help="Translate to language code")

    # Parse arguments
    return parser.parse_args()

def main():

    # Get command line arguments
    args = parse_args()

    # Check translation arguments
    if args.translate and not args.to_code:
        parser.error("Please specify target language --to_code with --translate")

    from_code = "en"

    if args.translate:
        # Translation setup if translation enabled
        to_code = args.to_code
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages))
        argostranslate.package.install_from_path(package_to_install.download())

    dataset_entries = []

    # Load input data
    with open(args.input) as f:
        data = json.load(f)

    # Process each item
    for item in data:
        # Only use higher quality outputs
        if item["source"] == "GPT-4":

            # Silent prompt tags
            prompt_start = "\u200B"
            prompt_end = "\u200C"

            if args.translate:
                # Translate story and summary if translation enabled
                story_trans = argostranslate.translate.translate(item["story"], from_code, to_code)
                story_summary = argostranslate.translate.translate(item["summary"], from_code, to_code)
            else:
                # Use original texts if no translation
                story_trans = item["story"]
                story_summary = item["summary"]

            if args.json:
                # Create dictionary for json output
                dataset_entry = {
                    "story": story_trans,
                    "summary": story_summary,
                    "language": to_code if args.translate else "en",
                }
                dataset_entries.append(dataset_entry)
            else:
                # Create concatenated dataset_entry string
                dataset_entry = f"\{prompt_start}{story_trans}\n\nPlease summarize the above.{prompt_end}\n\n{story_summary}\n\n"
                dataset_entries.append(dataset_entry)

    # Output dataset
    if args.output:
        with open(args.output, "w") as f:
            json.dump(dataset_entries, f)
    else:
        for dataset_entry in dataset_entries:
            print(dataset_entry)
            print()

if __name__ == "__main__":
    main()

