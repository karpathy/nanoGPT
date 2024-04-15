import os
from tqdm import tqdm
from datasets import load_dataset

def save_mnist_images(dataset, output_dir):
    """Saves images from the MNIST dataset to a specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Saving images"):
        image = item['image']  # The image is already a PIL.Image object.
        label = item['label']
        image.save(os.path.join(output_dir, f'{idx}_{label}.png'))

def main():
    # Load the MNIST dataset from Hugging Face
    dataset = load_dataset("mnist", split='train')

    # Specify the output directory for images
    output_dir = 'mnist_images'

    # Save images
    save_mnist_images(dataset, output_dir)
    print(f"Images saved in {output_dir}")

if __name__ == "__main__":
    main()

