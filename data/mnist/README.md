## MNIST Dataset and Preprocessing Scripts

This directory contains scripts compatible with the MNIST dataset, which is a
classic dataset of handwritten digits commonly used for training various image
processing systems and machine learning models, including Language Learning
Models (LLMs).

### Script `get_dataset.py`

The `get_dataset.py` script is used to download the MNIST dataset and save the
images to a specified directory.

**Usage:**
```bash
python3 get_dataset.py
```

By default, the script will download the MNIST dataset and save the images in
the `mnist_images` directory.

### Script `gray.py`

The `gray.py` script is used to convert images from the MNIST dataset into ASCII
art, which can be used for training LLMs or for visualization purposes.

**Usage:**

```bash
python3 gray.py --image-dir mnist_images --output-dimensions 8x8 --levels 2 --chars 01 --append-to-file
```

Options:
- `--image-dir` (required): Directory containing images to convert.
- `--output-dir` (default: `grayscale_images`): Directory to save ASCII art.
- `--output-dimensions` (default: `16x16`): Output dimensions for ASCII art, e.g., 8x8, 16x16.
- `--levels` (default: 2): Number of grayscale levels, currently 2 - 9 supported.
- `--chars` (optional): Custom characters for ASCII art, ordered from darkest to lightest.
- `--append-to-file` (optional): Append ASCII art to a single file instead of creating separate files.
- `--output-file` (default: `input.txt`): File to append ASCII art to if `--append-to-file` is used.
- `--number-placement` (default: `before`): Place the type of number before or after the ASCII image in the output file. Choices are `before` or `after`.

The script will process each image in the `--image-dir` directory and save the
resulting ASCII art to the `--output-dir` directory.

## License Information for MNIST Dataset

The MNIST dataset is made available under the terms of the [Creative Commons
Attribution-Share Alike 3.0
license](https://creativecommons.org/licenses/by-sa/3.0/). When using the MNIST
dataset, you should attribute the source as provided by the original authors.

Please note that while the MNIST dataset itself is licensed under the Creative
Commons license, the `get_dataset.py` and `gray.py` scripts provided in this
directory are subject to the license terms of the repository they reside in.

## Citation Information for MNIST Dataset

If you use the MNIST dataset in your research, please cite:

```bibtex
@article{lecun2010mnist,
  title={MNIST handwritten digit database},
  author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
  journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
  volume={2},
  year={2010}
}
```

