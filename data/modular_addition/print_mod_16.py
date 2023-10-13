import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base",
        help="base to use",
        type=int,
        choices=[1, 2, 4, 8, 16],
        default=16,
    )
    parser.add_argument("-s", "--seed", type=int, help="random seed")
    return parser.parse_args()


def to_base_4(num):
    if num == 0:
        return "0"

    digits = []
    while num > 0:
        remainder = num % 4
        digits.append(str(remainder))
        num //= 4

    return "".join(reversed(digits))


def convert_to_base(num, base):
    if base == 1:
        return "1" * num
    elif base == 2:
        return bin(num)[2:]
    elif base == 4:
        return to_base_4(num)
    elif base == 8:
        return oct(num)[2:]
    else:
        return hex(num)[2:]


def main(args):
    results = []
    for num1 in range(16):
        for num2 in range(16):
            result = (
                convert_to_base(num1, args.base),
                convert_to_base(num2, args.base),
                convert_to_base((num1 + num2) % 16, args.base),
            )
            results.append(result)

    if args.seed:
        random.seed(args.seed)

    random.shuffle(results)

    for result in results:
        print(" ".join(result))


if __name__ == "__main__":
    args = parse_args()
    main(args)
