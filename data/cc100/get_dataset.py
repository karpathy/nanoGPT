import requests
from tqdm import tqdm
import argparse
import lzma

LANGUAGE_CODES = [
    "af", "am", "ar", "as", "az", "be", "bg", "bn", "bn_rom", "br", "bs", "ca", "cs", "cy",
    "da", "de", "el", "en", "eo", "es", "et", "eu", "fa", "ff", "fi", "fr", "fy", "ga",
    "gd", "gl", "gn", "gu", "ha", "he", "hi", "hi_rom", "hr", "ht", "hu", "hy", "id", "ig",
    "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky", "la", "lg", "li",
    "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "my_zaw", "ne", "nl",
    "no", "ns", "om", "or", "pa", "pl", "ps", "pt", "qu", "rm", "ro", "ru", "sa", "si",
    "sc", "sd", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "ta_rom", "te",
    "te_rom", "th", "tl", "tn", "tr", "ug", "uk", "ur", "ur_rom", "uz", "vi", "wo", "xh",
    "yi", "yo", "zh-Hans", "zh-Hant", "zu", "all"
]

def download_file(url, filename):
    """
    Download a file from a given URL with a progress bar.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the download was successful.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Failed to download the file completely.")
    else:
        print(f"Downloaded {filename}")
    return filename

def extract_xz(file_path):
    """
    Extract an .xz file.
    """
    with lzma.open(file_path) as f:
        file_content = f.read()
    output_path = file_path[:-3]  # Remove '.xz' extension
    with open(output_path, 'wb') as f:
        f.write(file_content)
    print(f"Extracted {file_path} to {output_path}")

def main(language_code):
    base_url = "https://data.statmt.org/cc-100/"
    file_extension = ".txt.xz"
    
    if language_code == "all":
        for code in LANGUAGE_CODES[:-1]:  # Exclude "all"
            if code == "all":
                continue
            file_url = f"{base_url}{code}{file_extension}"
            output_filename = f"{code}{file_extension}"
            downloaded_file = download_file(file_url, output_filename)
            extract_xz(downloaded_file)
    else:
        file_url = f"{base_url}{language_code}{file_extension}"
        output_filename = f"{language_code}{file_extension}"
        downloaded_file = download_file(file_url, output_filename)
        extract_xz(downloaded_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract language-specific files."
    )
    parser.add_argument(
        "-c",
        "--code",
        type=str,
        choices=LANGUAGE_CODES,
        default="zh-Hans",
        help="Language code for the file to be downloaded or 'all' to download and extract all files.",
    )

    args = parser.parse_args()
    main(args.code)

