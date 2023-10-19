import math
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
    parser.add_argument(
        "-m",
        "--modulo",
        help="modulo to utilize",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--no_separator",
        action='store_true',
        help="do not separate numbers with spaces",
    )
    parser.add_argument(
        "--little_endian",
        action='store_true',
        help="reverse order of numbers",
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

def max_digits_modulo_in_base(digits, base):
    return math.ceil(math.log(digits, base))

def convert_to_base(num, base, digits):
    if base == 1:
        if num == 0:
          return str(int("0")).zfill(digits)[::-1]
        else:
          return str(int("1" * num)).zfill(digits)[::-1]
    elif base == 2:
        return str(bin(num)[2:].zfill(max_digits_modulo_in_base(digits, base)))[::-1]
    elif base == 4:
        return str(to_base_4(num).zfill(max_digits_modulo_in_base(digits, base)))[::-1]
    elif base == 8:
        return str(oct(num)[2:].zfill(max_digits_modulo_in_base(digits, base)))[::-1]
    else:
        return str(hex(num)[2:].zfill(max_digits_modulo_in_base(digits, base)))[::-1]



def main(args):
    results = []
    for num1 in range(args.modulo):
        for num2 in range(args.modulo):
            result = (
                convert_to_base(num1, args.base, args.modulo),
                convert_to_base(num2, args.base, args.modulo),
                convert_to_base((num1 + num2) % args.modulo, args.base, args.modulo),
            )
            results.append(result)

    if args.seed:
        random.seed(args.seed)

    random.shuffle(results)

    for result in results:
        if args.no_separator:
            print("".join(result))
        else:
            print(" ".join(result))


if __name__ == "__main__":
    args = parse_args()
    main(args)
