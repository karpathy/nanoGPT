# Rubik's Cube Simulator

## Overview
This project is a Rubik's Cube simulator implemented in Python. It allows you to
simulate the operations and solution algorithms for the Rubik's Cube, providing
options for shuffling the cube, applying moves, and printing the cube state in
different formats.

## Features
- **Shuffle**: Randomly shuffle the Rubik's Cube with a specified number of moves.
- **Moves**: Apply a specified number of moves to the cube and print each move.
- **Output**: Print the cube's state to the console or an optional output file.
- **Formats**: Print the cube in a condensed or 'unfolded' cube format.

## Usage

### Arguments
- `-s, --shuffle`: Number of random moves to shuffle the cube before starting to print (default: 0).
- `-m, --moves`: Number of moves to print to the stdout (default: 1).
- `-o, --output`: Optional output file to use instead of stdout.
- `-c, --condensed`: Optional condensed form without spaces.
- `-p, --prefix`: Prefix to place before each move type (default: "m").

### Example Commands

#### Shuffle and Print Moves to Console

Shuffle 20 random moves first before printing 5 moves to stdout
```sh
python rubiks_cube.py -s 10 -m 5
```

#### Shuffle and Print to an Output File in Condensed Format

Shuffle 20 random moves first before printing 10 moves to stdout
Utilize condensed Format
Use "@" as the prefix for the move type.

```sh
python rubiks_cube.py -s 20 -m 10 -o output.txt -c -p "@"
```

### Sample Output

#### Unfolded Cube Format (Default)

This format is a little easier to read, centering attention on the green side.
```
      W W W
      W W W
      W W W
O O O G G G R R R B B B
O O O G G G R R R B B B
O O O G G G R R R B B B
      Y Y Y
      Y Y Y
      Y Y Y
```

#### Condensed Format

This format is more efficient for training models.
This reduces the number of spaces, removing inbetween faces and reorganizing
the representation to remove the need for leading spaces.

```
WWW
WWW
WWW
GGGRRRBBBOOO
GGGRRRBBBOOO
GGGRRRBBBOOO
YYY
YYY
YYY
```

