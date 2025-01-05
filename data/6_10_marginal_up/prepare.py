# Dataset File Name: train.txt
# Download file from huggingface if file does not exist
import os, re
# Same directory as this file
cur_path = os.path.dirname(os.path.realpath(__file__))
train_file_path = os.path.join(cur_path, "train.txt")
test_file_path = os.path.join(cur_path, "test.txt")

train_data_url = "https://huggingface.co/datasets/leyanpan/sat-solver/resolve/main/large-500k/SAT_6_10_marginal_large.txt?download=true"
test_data_url = "https://huggingface.co/datasets/leyanpan/sat-solver/resolve/main/SAT_6_10_random_Test.txt?download=true"

def remove_states_followed_by_up(s: str) -> str:
    """
    Remove all states followed by the [UP] token and the [UP] token themselves.
    """
    separators = r'\[SEP\]|\[BT\]|\[UP\]'
    parts = re.split(f'({separators})', s)
    output = []
    i = 0

    while i < len(parts):
        part = parts[i]
        if part == '[UP]':
            if output and re.match(separators, output[-1]) is None:
                output.pop()
        elif part.strip() != '[UP]':
            output.append(part)
        i += 1

    return ''.join(output).strip()


if not os.path.exists(train_file_path):
    print("Downloading training dataset...")
    import urllib.request

    urllib.request.urlretrieve(train_data_url, train_file_path)
    print("Dataset downloaded.")

    with open(train_file_path, 'r') as f:
        lines = f.readlines()

if not os.path.exists(test_file_path):
    print("Downloading test dataset...")
    import urllib.request

    urllib.request.urlretrieve(test_data_url, test_file_path)
    print("Dataset downloaded.")