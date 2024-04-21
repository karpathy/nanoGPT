import argparse
from rich.progress import Progress, BarColumn, TextColumn
import time
import random
from datetime import datetime, timedelta
import os



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

def test_all_additions(max_digits, output_file, random_mode, max_examples_per_digit):
    start_time = time.time()
    correct = 0
    processed = 0

    # Prepare the range of numbers and limit them as per the max_examples_per_digit
    ranges = []
    for num_digits in range(1, max_digits + 1):
        number_range = range(10**(num_digits-1), 10**num_digits)
        if max_examples_per_digit:
            number_range = random.sample(list(number_range), min(len(number_range), max_examples_per_digit))
        for a in range(1, 10):
            for b in number_range:
                ranges.append((a, b))
    
    if random_mode:
        ranges = random.sample(ranges, len(ranges))  # Randomly shuffle the entire list

    total_tests = len(ranges)

    with open(output_file, 'w') as file, Progress(
        TextColumn("[bold blue]{task.fields[eta]}"),
        TextColumn("[bold blue]Completion: {task.fields[completion_time]}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}% Complete"),
        TextColumn("[bold green]{task.fields[mb]} MB"),
        TextColumn("[bold green]Final Size Estimate: {task.fields[final_mb_estimate]} MB")
    ) as progress:
        task = progress.add_task("Testing additions...", total=total_tests, eta="Calculating...",
                                 completion_time="Calculating...", mb="0.00 MB", final_mb_estimate="Calculating...")

        for a, b in ranges:
            script_output, script_result = detailed_addition(a, b)
            correct_result = a + b

            for line in script_output:
                file.write(line + '\n')
            file.flush()
            current_size = os.path.getsize(output_file) / (1024 * 1024)
            processed += 1

            update_progress(progress, task, processed, total_tests, start_time, current_size)

            if script_result != correct_result:
                print(f"Error in addition a={a} and b={b}: Expected {correct_result}, got {script_result}")
            else:
                correct += 1

        progress.remove_task(task)
    print(f"Number of correct results: {correct}/{total_tests}")


def update_progress(progress, task, processed, total_tests, start_time, current_size):
    elapsed_time = time.time() - start_time
    avg_time_per_test = elapsed_time / processed if processed > 0 else 0
    eta_seconds = int(avg_time_per_test * (total_tests - processed))
    eta = f"{eta_seconds // 60}m {eta_seconds % 60}s" if processed > 0 else "Calculating..."

    completion_time = datetime.now() + timedelta(seconds=eta_seconds)
    completion_time_str = completion_time.strftime('%Y-%m-%d %H:%M:%S')
    final_mb_estimate = (current_size / processed) * total_tests if processed > 0 else 0

    progress.update(task, advance=1, mb=f"{current_size:.2f}", eta=eta,
                    completion_time=completion_time_str, final_mb_estimate=f"{final_mb_estimate:.2f} MB")



def main():
    parser = argparse.ArgumentParser(description="Test all possible additions for specified digit lengths.")
    parser.add_argument("--digits", type=int, default=2, help="Number of digits for the second number b")
    parser.add_argument("--output_file", type=str, default="input.txt", help="Output file name for results")
    parser.add_argument("--random", action="store_true", help="Enable random selection of number combinations")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum examples per digit length")

    args = parser.parse_args()
    test_all_additions(args.digits, args.output_file, args.random, args.max_examples)

if __name__ == "__main__":
    main()
