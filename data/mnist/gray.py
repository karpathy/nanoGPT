from PIL import Image
import argparse
import os
import numpy as np
import sys

def convert_image_to_ascii(image_path, output_size, grayscale, levels, chars=None):
    # Load the image from the specified path and resize it
    img = Image.open(image_path).resize(output_size, Image.LANCZOS)

    if grayscale:
        img = img.convert("L")  # Convert to grayscale
        if levels < 256:
            img = img.point(lambda p: (p * levels) // 256 * (256 // levels))

    # Define default characters for different levels
    default_chars = "@%#*+=-:. "
    if chars is None:
        if levels == 2:
            chars = "@ "
        elif levels == 3:
            chars = "@. "
        elif levels == 4:
            chars = "@*- "
        elif levels == 5:
            chars = "@#+- "
        elif levels == 6:
            chars = "@#+-. "
        elif levels == 7:
            chars = "@#+-:. "
        elif levels == 8:
            chars = "@#*+-:. "
        elif levels == 9:
            chars = "@%#*+-:. "
        else:
            sys.exit(f"number of levels {levels} not supported")

    # Normalize the characters set based on the number of levels
    char_array = np.array([c for c in chars])
    n_chars = len(char_array)
    scale_factor = 256 // levels

    # Convert the image to a numpy array
    img_np = np.array(img)

    # Convert each pixel to the corresponding ASCII character
    ascii_img = char_array[img_np // scale_factor]

    # Join characters to form lines and then join lines to form the full ASCII image
    ascii_img_lines = ["".join(row) for row in ascii_img]
    ascii_result = "\n".join(ascii_img_lines)

    return ascii_result

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert images to ASCII art.')
    parser.add_argument('--output-dimensions', type=str, default='16x16', help='Output dimensions for ASCII art, e.g., 8x8, 16x16')
    parser.add_argument('--levels', type=int, default=2, help='Number of grayscale levels, currently 2 - 9 supported')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing images to convert.')
    parser.add_argument('--output-dir', type=str, default='grayscale_images', help='Directory to save ASCII art.')
    parser.add_argument('--append-to-file', action='store_true', help='Append ASCII art to a single file.')
    parser.add_argument('--output-file', type=str, default='input.txt', help='File to append ASCII art to.')
    parser.add_argument('--number-placement', type=str, default='before', choices=['before', 'after'], help='Place the type of number before or after the ASCII image in the output file.')
    parser.add_argument('--chars', type=str, default=None, help='Custom characters for ASCII art, ordered from darkest to lightest.')
    args = parser.parse_args()

    # Parse output dimensions
    output_dimensions = tuple(map(int, args.output_dimensions.split('x')))

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Open the output file for appending if required
    output_file = None
    if args.append_to_file:
        output_file = open(args.output_file, 'a')

    # Process each image in the directory
    for image_filename in os.listdir(args.image_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(args.image_dir, image_filename)
            ascii_art = convert_image_to_ascii(image_path, output_size=output_dimensions, grayscale=True, levels=args.levels, chars=args.chars)
            if args.append_to_file:
                # Determine the number from the filename
                number = os.path.splitext(image_filename)[0].split('_')[-1]
                # Append the ASCII art to the output file with the number placement
                if args.number_placement == 'before':
                    output_file.write(f'{number}\n{ascii_art}\n')
                else:
                    output_file.write(f'{ascii_art}\n{number}\n')
            else:
                # Save the ASCII art to a text file
                output_path = os.path.join(args.output_dir, os.path.splitext(image_filename)[0] + '.txt')
                with open(output_path, 'w') as f:
                    f.write(ascii_art)
                print(f'ASCII art saved to {output_path}')

    # Close the output file if it was opened
    if output_file:
        output_file.close()
