import os
import pickle
import requests
import numpy as np

# Define the path for the input file
#input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

import requests
import subprocess
# The command to be executed
command = "wget https://mattmahoney.net/dc/enwik8.zip"

# Simulate issuing the command (this will not execute in this environment)
try:
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    error = result.stderr.decode('utf-8')
    success = True
except Exception as e:
    output = ""
    error = str(e)
    success = False

print(success, output, error)

# unzip
command = "unzip enwik8.zip"
try:
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    error = result.stderr.decode('utf-8')
    success = True
except Exception as e:
    output = ""
    error = str(e)
    success = False

print(success, output, error)



# Read the dataset as bytes
with open('enwik8', 'rb') as file:
    data = file.read()
print(f"length of dataset in characters: {len(data):,}")

# Get all the unique bytes that occur in this text
bytes_set = sorted(set(data))
vocab_size = len(bytes_set)
print("all the unique bytes:", bytes_set)
print(f"vocab size: {vocab_size:,}")

# Create a mapping from bytes to integers and integers to bytes
stoi = {byte: i for i, byte in enumerate(bytes_set)}
itos = {i: byte for i, byte in enumerate(bytes_set)}

def encode(s):
    # Encoder: take a bytes object, output a list of integers
    return [stoi[b] for b in s]

def decode(l):
    # Decoder: take a list of integers, output a bytes object
    return bytes([itos[i] for i in l])

# Create the train and test splits

# Split the data into training, validation, and testing sets
n = len(data)
train_end = int(n * 0.90)
val_end = train_end + int(n * 0.05)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# Encode the data to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
test_ids = encode(test_data)

# Convert to numpy arrays for efficiency
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

# Define the directory to save the processed files
output_dir = '.'  # You can change this to your desired path

# Save the splits to binary files
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))
test_ids.tofile(os.path.join(output_dir, 'test.bin'))

# Save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as file:
    pickle.dump(meta, file)