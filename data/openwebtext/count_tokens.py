import numpy as np
import os

def count_tokens(file_path):
    try:
        # Open the binary file using np.memmap
        data = np.memmap(file_path, dtype=np.uint16, mode='r')
        
        # The number of tokens is the length of the memmap array
        num_tokens = len(data)
        
        return num_tokens
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    # Path to the train.bin file
    train_bin_path = 'train.bin'
    
    # Ensure the file exists
    if not os.path.exists(train_bin_path):
        print(f"File not found: {train_bin_path}")
    else:
        num_tokens = count_tokens(train_bin_path)
        if num_tokens is not None:
            print(f"Number of tokens in {train_bin_path}: {num_tokens}")
