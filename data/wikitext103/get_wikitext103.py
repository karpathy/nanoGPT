import torchtext
from torchtext.datasets import WikiText103

def download_wikitext103():
    # Define the dataset path
    dataset_path = '.data'

    # Download the WikiText-103 dataset
    train_iter, valid_iter, test_iter = WikiText103(root=dataset_path, split=('train', 'valid', 'test'))

    # Open a file to save the dataset
    with open('wikitext103.txt', 'w') as file:
        for dataset in [train_iter, valid_iter, test_iter]:
            for line in dataset:
                file.write(line + '\n')

if __name__ == "__main__":
    download_wikitext103()

