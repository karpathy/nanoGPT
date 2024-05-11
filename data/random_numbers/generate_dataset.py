import argparse
import random
import sys
from tqdm import tqdm

def generate_random_numbers(base, count):
    try:
        if base < 2:
            raise ValueError("Base must be an integer greater than or equal to 2.")
    except ValueError:
        raise ValueError("Base must be an integer.")

    return ''.join(format(random.randint(0, base - 1), 'x') for _ in tqdm(range(count), desc="Generating random numbers"))

def main():
    parser = argparse.ArgumentParser(description='Generate a dataset of random numbers.')
    parser.add_argument('--base', type=int, required=True,
                        help='The base for random number generation. Must be an integer greater than or equal to 2.')
    parser.add_argument('--count', type=int, required=True,
                        help='The number of random tokens to generate.')
    parser.add_argument('--seed', type=int, default=None,
                        help='The seed for the random number generator.')
    parser.add_argument('--output', type=str, default='input.txt',
                        help='The output file to save the random numbers.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    try:
        result = generate_random_numbers(args.base, args.count)
        with open(args.output, 'w') as file:
            file.write(result)
    except ValueError as e:
        sys.stderr.write(str(e))
        sys.exit(1)

if __name__ == '__main__':
    main()
