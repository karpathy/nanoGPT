import argparse

def detailed_addition(a, b):
    a = str(a)
    b = str(b)
    # Correctly pad 'b' to have one more digit than its current length
    padded_b = '0' + b

    result = []
    carry = 0
    max_length = len(padded_b)

    outputs = []
    outputs.append(f"s:a_{a}d0b_{'_'.join(padded_b)}c0e0")

    for i in range(max_length):
        digit_a = int(a) if i == 0 else '_'+a
        active_a = f"a>{a}" if i == 0 else f"a{digit_a}"
        digit_b = int(padded_b[max_length - 1 - i])
        sum_digits = (int(a) if i == 0 else 0) + digit_b + carry

        new_carry = sum_digits // 10
        current_digit = sum_digits % 10
        result.append(current_digit)

        active_b_parts = list(padded_b)
        active_b_parts[max_length - 1 - i] = '>' + active_b_parts[max_length - 1 - i]
        active_b = '_'.join(active_b_parts)

        previous_result = str(carry)
        formatted_result = '^'.join(str(x) for x in reversed(result)) if i > 0 else str(current_digit)
        formatted_result_e = '^' + formatted_result.replace('^', '_')
        outputs.append(f"m:{active_a}d>{previous_result}b{active_b}c{new_carry}e{formatted_result_e}")

        carry = new_carry

    if carry > 0:
        result.append(carry)
    final_result = ''.join(str(x) for x in reversed(result))
    outputs.append(f"r:a{digit_a}d{previous_result}b{padded_b}c0e{final_result}")

    return outputs, int(final_result)

def detailed_addition(a, b):
    a = str(a)
    b = str(b)
    padded_b = '0' + b
    result = []
    carry = 0
    max_length = len(padded_b)

    outputs = []
    outputs.append(f"s:a_{a}d0b_{'_'.join(padded_b)}c0e0")

    for i in range(max_length):
        digit_a = int(a) if i == 0 else '_'+a
        active_a = f"a>{a}" if i == 0 else f"a{digit_a}"
        digit_b = int(padded_b[max_length - 1 - i])
        sum_digits = (int(a) if i == 0 else 0) + digit_b + carry

        new_carry = sum_digits // 10
        current_digit = sum_digits % 10
        result.append(current_digit)

        active_b_parts = list(padded_b)
        active_b_parts[max_length - 1 - i] = '>' + active_b_parts[max_length - 1 - i]
        active_b = '_'.join(active_b_parts)

        previous_result = str(carry)
        formatted_result = '^'.join(str(x) for x in reversed(result)) if i > 0 else str(current_digit)
        formatted_result_e = '^' + formatted_result.replace('^', '_')
        outputs.append(f"m:{active_a}d>{previous_result}b{active_b}c{new_carry}e{formatted_result_e}")

        carry = new_carry

    if carry > 0:
        result.append(carry)
    final_result = ''.join(str(x) for x in reversed(result))
    outputs.append(f"r:a{digit_a}d{previous_result}b{padded_b}c0e{final_result}")

    return outputs, int(final_result)


def test_all_additions(max_digits, output_file):
    correct = 0
    total_tests = 0
    with open(output_file, 'w') as file:
        for a in range(1, 10):  # Single-digit 'a' from 1 to 9
            for num_digits in range(1, max_digits + 1):  # Test for 1 to max_digits lengths of 'b'
                for b in range(10**(num_digits-1), 10**num_digits):  # 'b' from the lowest to the highest with `num_digits`
                    script_output, script_result = detailed_addition(a, b)
                    correct_result = a + b
                    total_tests += 1

                    for line in script_output:
                        file.write(line + '\n')
                    if script_result == correct_result:
                        correct += 1
                    else:
                        print(f"Error in addition a={a} and b={b}: Expected {correct_result}, got {script_result}")

    print(f"Number of correct results: {correct}/{total_tests}")
def main():
    parser = argparse.ArgumentParser(description="Test all possible additions for specified digit lengths.")
    parser.add_argument("--digits", type=int, default=2, help="Number of digits for the second number b")
    parser.add_argument("--output_file", type=str, default="input.txt", help="Output file name for results")

    args = parser.parse_args()
    test_all_additions(args.digits, args.output_file)

if __name__ == "__main__":
    main()

