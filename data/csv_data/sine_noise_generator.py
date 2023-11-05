import argparse
import numpy as np
import pandas as pd

# Function to generate sine wave with noise
def generate_sine_wave_with_noise(freq, sample_rate, num_points, noise_level):
    t = np.arange(num_points)  # Time axis with integer values starting from 0
    sine_wave = np.sin(2 * np.pi * freq * t / sample_rate)
    noise = noise_level * np.random.normal(size=num_points)
    combined_signal = sine_wave + noise
    return t, sine_wave, noise, combined_signal

# Function to format data as scientific notation if required
def format_data_as_scientific(df, precision):
    return df.applymap(lambda x: f"{x:.{precision}e}")

# Function to save the data to CSV
def save_to_csv(time, signal, noise, combined_signal, filename, scientific, precision, split_files, modulo):
    # Apply modulo if necessary
    if modulo is not None:
        time = time % modulo

    # Create the DataFrame
    data = {
        'signal': signal,
        'noise': noise,
        'signal_plus_noise': combined_signal
    }

    if split_files:
        # Save time data to a separate CSV
        time_df = pd.DataFrame({'seconds_from_start': time})
        if scientific:
            time_df = format_data_as_scientific(time_df, precision)
        time_df.to_csv(f"time_{filename}", header=False, index=False)

        # Save data to a separate CSV
        data_df = pd.DataFrame(data)
        if scientific:
            data_df = format_data_as_scientific(data_df, precision)
        data_df.to_csv(f"data_{filename}", header=False, index=False)
    else:
        # Combine time and data for a single CSV
        df = pd.DataFrame({'seconds_from_start': time, **data})
        if scientific:
            df = format_data_as_scientific(df, precision)
        df.to_csv(filename, header=False, index=False)

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate a sine wave with noise and export to CSV.')
    parser.add_argument('-n', '--noise_level', type=float, default=0.5, required=True,
                        help='Level of noise relative to the sine wave (0-1).')
    parser.add_argument('-f', '--filename', type=str, default='sine_wave.csv',
                        help='Name of the output CSV file.')
    parser.add_argument('--scientific', action='store_true',
                        help='Output numbers in scientific notation.')
    parser.add_argument('--precision', type=int, default=2,
                        help='Number of digits past the decimal point in scientific notation.')
    parser.add_argument('--points', type=int, default=5000,
                        help='Total number of data points to be created.')
    parser.add_argument('--split', action='store_true',
                        help='Save time data and signal data in separate CSV files.')
    parser.add_argument('--modulo', type=int,
                        help='Modulo value to apply to the time data.')
    args = parser.parse_args()
    if not (0 <= args.noise_level <= 1):
        raise ValueError('Noise level must be between 0 and 1.')
    if args.precision < 0:
        raise ValueError('Precision must be a non-negative integer.')
    if args.points <= 0:
        raise ValueError('Number of data points must be a positive integer.')
    return args

def main():
    args = parse_arguments()

    # Parameters for sine wave generation
    frequency = 5  # Frequency in Hz
    sample_rate = 500  # Sample rate in Hz
    num_points = args.points  # Total number of data points

    # Generate the sine wave with noise
    time, sine_wave, noise, combined_signal = generate_sine_wave_with_noise(
        frequency, sample_rate, num_points, args.noise_level
    )

    # Save to CSV file(s)
    save_to_csv(time, sine_wave, noise, combined_signal, args.filename, args.scientific, args.precision, args.split, args.modulo)
    if args.split:
        print(f"Time data saved to time_{args.filename}")
        print(f"Signal data saved to data_{args.filename}")
    else:
        print(f"Sine wave data with noise saved to {args.filename}")

if __name__ == '__main__':
    main()

