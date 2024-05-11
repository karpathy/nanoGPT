# Dataset Folder README

## Creating a new dataset

To speed up creation of a new dataset, initialize it with the following script.

This will create the necessary directory structure and symlink common utility
scripts for dataset preparation.

```sh
bash create_new_dataset.sh <name-of-dataset>
```

## Combining Datasets

To combine binary dataset files from multiple directories into single train and
validation files, use the `combine_datasets.py` script.

This is useful when you want to merge data from different sources.

```sh
python combine_datasets.py --dirs <list-of-directories> --output_dir <output-directory>
```

## Wishlist

- [ ] Custom phoneme-token-list per language.
- [ ] Script to merge phoneme lists.

