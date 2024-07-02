import numpy as np
import random

# Define the mapping dictionaries
letter_mapping = {
    'H': (0.866, 0.5),
    'M': (1.0, 0.0),
    'L': (0.866, -0.5),
    'y': (0.866, 0.5),
    'n': (0.866, -0.5),
    's': (0.866, 0.5),
    'a': (1.0, 0.0),
    'f': (0.866, -0.5),
}

# Define a function to generate random coordinates
def random_coordinates(mean=0.0, stdev=0.02):
    return (random.gauss(mean, stdev), random.gauss(mean, stdev))

# Define a function to map letters
def map_letter(letter):
    if letter in letter_mapping:
        return letter_mapping[letter]
    elif letter == 'r':
        return random_coordinates()
    else:
        # For unspecified letters, return random coordinates
        return random_coordinates()

# # Input table
# table = [
#     ['a', 'L', 'L', 'y', 'n', 'n', 'y', 'L', 'n', 'n', 'n', 'n', 'r'],
#     ['i', 'H', 'L', 'y', 'n', 'n', 'y', 'L', 'n', 'y', 'n', 'n', 'r'],
#     ['u', 'L', 'H', 'y', 'y', 'n', 'y', 'L', 'y', 'n', 'n', 'n', 'r'],
#     ['e', 'M', 'M', 'y', 'n', 'n', 'y', 'L', 'n', 'y', 'n', 'n', 'r'],
#     ['o', 'L', 'M', 'y', 'y', 'n', 'y', 'L', 'y', 'n', 'n', 'n', 'r'],
#     ['k', 'L', 'H', 'y', 'n', 'n', 'n', 'H', 'y', 'n', 'n', 'n', 's'],
#     ['s', 'H', 'L', 'y', 'n', 'n', 'n', 'M', 'n', 'n', 'y', 'n', 'r'],
#     ['t', 'H', 'L', 'y', 'n', 'n', 'n', 'H', 'n', 'n', 'y', 'n', 's'],
#     ['x', 'H', 'L', 'y', 'n', 'n', 'n', 'H', 'n', 'y', 'n', 'n', 'a'],
#     ['n', 'H', 'L', 'y', 'n', 'n', 'y', 'L', 'n', 'n', 'y', 'n', 'r'],
#     ['h', 'M', 'L', 'y', 'n', 'n', 'n', 'H', 'n', 'n', 'n', 'n', 'r'],
#     ['m', 'L', 'L', 'n', 'n', 'n', 'y', 'L', 'n', 'n', 'n', 'n', 'r'],
#     ['y', 'M', 'H', 'y', 'n', 'n', 'y', 'L', 'n', 'n', 'n', 'n', 'r'],
#     ['r', 'H', 'L', 'y', 'n', 'n', 'y', 'L', 'n', 'n', 'y', 'y', 'r'],
#     ['w', 'M', 'H', 'y', 'y', 'n', 'y', 'L', 'n', 'n', 'n', 'n', 'r'],
#     ['g', 'L', 'H', 'y', 'n', 'n', 'y', 'M', 'y', 'n', 'n', 'n', 's'],
#     ['r', 'H', 'L', 'y', 'n', 'n', 'y', 'M', 'n', 'n', 'y', 'n', 'r'],
#     ['d', 'H', 'L', 'y', 'n', 'n', 'y', 'M', 'y', 'n', 'n', 'n', 's'],
#     ['b', 'M', 'L', 'n', 'n', 'y', 'y', 'M', 'n', 'n', 'n', 'n', 's'],
#     ['p', 'M', 'L', 'n', 'n', 'y', 'y', 'H', 'n', 'n', 'n', 'n', 's'],
#     ['sh', 'L', 'H', 'y', 'y', 'n', 'n', 'n', 'H', 'y', 'n', 'n', 'f'],
#     ['j', 'L', 'L', 'y', 'n', 'n', 'y', 'L', 'n', 'n', 'n', 'n', 'a'],
#     ['_', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
# ]
# Input table
table = [
    ['_', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    ['a', 'L', 'L', 'y', 'n', 'n', 'y', 'L', 'n', 'n', 'n', 'n', 'r'],
    ['b', 'M', 'L', 'n', 'n', 'y', 'y', 'M', 'n', 'n', 'n', 'n', 's'],
    ['x', 'H', 'L', 'y', 'n', 'n', 'n', 'H', 'n', 'y', 'n', 'n', 'a'],
    ['d', 'H', 'L', 'y', 'n', 'n', 'y', 'M', 'y', 'n', 'n', 'n', 's'],
    ['e', 'M', 'M', 'y', 'n', 'n', 'y', 'L', 'n', 'y', 'n', 'n', 'r'],
    ['g', 'L', 'H', 'y', 'n', 'n', 'y', 'M', 'y', 'n', 'n', 'n', 's'],
    ['h', 'M', 'L', 'y', 'n', 'n', 'n', 'H', 'n', 'n', 'n', 'n', 'r'],
    ['i', 'H', 'L', 'y', 'n', 'n', 'y', 'L', 'n', 'y', 'n', 'n', 'r'],
    ['j', 'L', 'L', 'y', 'n', 'n', 'y', 'L', 'n', 'n', 'n', 'n', 'a'],
    ['k', 'L', 'H', 'y', 'n', 'n', 'n', 'H', 'y', 'n', 'n', 'n', 's'],
    ['m', 'L', 'L', 'n', 'n', 'n', 'y', 'L', 'n', 'n', 'n', 'n', 'r'],
    ['n', 'H', 'L', 'y', 'n', 'n', 'y', 'L', 'n', 'n', 'y', 'n', 'r'],
    ['o', 'L', 'M', 'y', 'y', 'n', 'y', 'L', 'y', 'n', 'n', 'n', 'r'],
    ['p', 'M', 'L', 'n', 'n', 'y', 'y', 'H', 'n', 'n', 'n', 'n', 's'],
    ['r', 'H', 'L', 'y', 'n', 'n', 'y', 'L', 'n', 'n', 'y', 'y', 'r'],
    ['r', 'H', 'L', 'y', 'n', 'n', 'y', 'M', 'n', 'n', 'y', 'n', 'r'],
    ['s', 'H', 'L', 'y', 'n', 'n', 'n', 'M', 'n', 'n', 'y', 'n', 'r'],
    ['sh', 'L', 'H', 'y', 'y', 'n', 'n', 'n', 'H', 'y', 'n', 'n', 'f'],
    ['t', 'H', 'L', 'y', 'n', 'n', 'n', 'H', 'n', 'n', 'y', 'n', 's'],
    ['u', 'L', 'H', 'y', 'y', 'n', 'y', 'L', 'y', 'n', 'n', 'n', 'r'],
    ['w', 'M', 'H', 'y', 'y', 'n', 'y', 'L', 'n', 'n', 'n', 'n', 'r'],
    ['y', 'M', 'H', 'y', 'n', 'n', 'y', 'L', 'n', 'n', 'n', 'n', 'r'],
]



# Map the table
mapped_table = [[item for letter in row for item in map_letter(letter)] for row in table]

# Convert to numpy array
wte = np.array(mapped_table)

# Print the shape of the wte
print(f"Shape of wte: {wte.shape}")

# Save the wte as a .npy file
np.save('initial_wte.npy', wte)

print(f"Saved initial wte with shape {wte.shape} to initial_wte.npy")

# Print the first few rows of the wte with 3 decimal places
print("\nPrint wte (3 decimal places):")
np.set_printoptions(precision=3, suppress=True)
print(wte)

