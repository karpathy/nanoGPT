import argparse
import random
import sys
from tqdm import tqdm

class RubiksCube:
    def __init__(self, condensed_output=False):
        # whether to print condensed form or cross format
        self.condensed = condensed_output

        self.faces = {
            'U': [['W'] * 3 for _ in range(3)],
            'D': [['Y'] * 3 for _ in range(3)],
            'F': [['G'] * 3 for _ in range(3)],
            'B': [['B'] * 3 for _ in range(3)],
            'L': [['O'] * 3 for _ in range(3)],
            'R': [['R'] * 3 for _ in range(3)]
        }
        self.moves = {
            'a': self.rotate_U,
            'v': self.rotate_D,
            'c': self.rotate_F,
            'd': self.rotate_B,
            'e': self.rotate_L,
            'f': self.rotate_R,
            'A': self.rotate_U_inv,
            'V': self.rotate_D_inv,
            'C': self.rotate_F_inv,
            'D': self.rotate_B_inv,
            'E': self.rotate_L_inv,
            'F': self.rotate_R_inv,
        }

    def rotate_face_clockwise(self, face):
        return [list(row) for row in zip(*face[::-1])]

    def rotate_face_counterclockwise(self, face):
        return [list(row) for row in zip(*face)][::-1]

    def rotate_U(self):
        self.faces['U'] = self.rotate_face_clockwise(self.faces['U'])
        self.faces['F'][0], self.faces['R'][0], self.faces['B'][0], self.faces['L'][0] = \
            self.faces['R'][0], self.faces['B'][0], self.faces['L'][0], self.faces['F'][0]

    def rotate_U_inv(self):
        self.faces['U'] = self.rotate_face_counterclockwise(self.faces['U'])
        self.faces['F'][0], self.faces['L'][0], self.faces['B'][0], self.faces['R'][0] = \
            self.faces['L'][0], self.faces['B'][0], self.faces['R'][0], self.faces['F'][0]

    def rotate_D(self):
        self.faces['D'] = self.rotate_face_clockwise(self.faces['D'])
        self.faces['F'][2], self.faces['L'][2], self.faces['B'][2], self.faces['R'][2] = \
            self.faces['L'][2], self.faces['B'][2], self.faces['R'][2], self.faces['F'][2]

    def rotate_D_inv(self):
        self.faces['D'] = self.rotate_face_counterclockwise(self.faces['D'])
        self.faces['F'][2], self.faces['R'][2], self.faces['B'][2], self.faces['L'][2] = \
            self.faces['R'][2], self.faces['B'][2], self.faces['L'][2], self.faces['F'][2]

    def rotate_F(self):
        self.faces['F'] = self.rotate_face_clockwise(self.faces['F'])
        for i in range(3):
            self.faces['U'][2][i], self.faces['R'][i][0], self.faces['D'][0][2 - i], self.faces['L'][2 - i][2] = \
                self.faces['L'][2 - i][2], self.faces['U'][2][i], self.faces['R'][i][0], self.faces['D'][0][2 - i]

    def rotate_F_inv(self):
        self.faces['F'] = self.rotate_face_counterclockwise(self.faces['F'])
        for i in range(3):
            self.faces['U'][2][i], self.faces['L'][2 - i][2], self.faces['D'][0][2 - i], self.faces['R'][i][0] = \
                self.faces['R'][i][0], self.faces['U'][2][i], self.faces['L'][2 - i][2], self.faces['D'][0][2 - i]

    def rotate_B(self):
        self.faces['B'] = self.rotate_face_clockwise(self.faces['B'])
        for i in range(3):
            self.faces['U'][0][i], self.faces['L'][2 - i][0], self.faces['D'][2][2 - i], self.faces['R'][i][2] = \
                self.faces['R'][i][2], self.faces['U'][0][i], self.faces['L'][2 - i][0], self.faces['D'][2][2 - i]

    def rotate_B_inv(self):
        self.faces['B'] = self.rotate_face_counterclockwise(self.faces['B'])
        for i in range(3):
            self.faces['U'][0][i], self.faces['R'][i][2], self.faces['D'][2][2 - i], self.faces['L'][2 - i][0] = \
                self.faces['L'][2 - i][0], self.faces['U'][0][i], self.faces['R'][i][2], self.faces['D'][2][2 - i]

    def rotate_L(self):
        self.faces['L'] = self.rotate_face_clockwise(self.faces['L'])
        for i in range(3):
            self.faces['U'][i][0], self.faces['F'][i][0], self.faces['D'][i][0], self.faces['B'][2 - i][2] = \
                self.faces['B'][2 - i][2], self.faces['U'][i][0], self.faces['F'][i][0], self.faces['D'][i][0]

    def rotate_L_inv(self):
        self.faces['L'] = self.rotate_face_counterclockwise(self.faces['L'])
        for i in range(3):
            self.faces['U'][i][0], self.faces['B'][2 - i][2], self.faces['D'][i][0], self.faces['F'][i][0] = \
                self.faces['F'][i][0], self.faces['U'][i][0], self.faces['B'][2 - i][2], self.faces['D'][i][0]

    def rotate_R(self):
        self.faces['R'] = self.rotate_face_clockwise(self.faces['R'])
        for i in range(3):
            self.faces['U'][i][2], self.faces['B'][2 - i][0], self.faces['D'][i][2], self.faces['F'][i][2] = \
                self.faces['F'][i][2], self.faces['U'][i][2], self.faces['B'][2 - i][0], self.faces['D'][i][2]

    def rotate_R_inv(self):
        self.faces['R'] = self.rotate_face_counterclockwise(self.faces['R'])
        for i in range(3):
            self.faces['U'][i][2], self.faces['F'][i][2], self.faces['D'][i][2], self.faces['B'][2 - i][0] = \
                self.faces['B'][2 - i][0], self.faces['U'][i][2], self.faces['F'][i][2], self.faces['D'][i][2]

    def shuffle(self, k):
        moves = list(self.moves.keys())
        for _ in tqdm(range(k), desc="Shuffling"):
            move = random.choice(moves)
            self.moves[move]()

    def print_cube(self, output):
        def print_face(face):
            return '\n'.join(''.join(row) for row in face)

        if self.condensed:
            output.write(print_face(self.faces['U']) + "\n")
            for i in range(3):
                output.write(''.join(self.faces['F'][i]) + '' + ''.join(self.faces['R'][i]) + '' + ''.join(self.faces['B'][i]) + ''.join(self.faces['L'][i]) + "\n")
            output.write(print_face(self.faces['D']) + "\n")
        else:
            output.write("    " + print_face(self.faces['U']).replace('\n', '\n    ') + "\n")
            for i in range(3):
                output.write(' '.join(self.faces['L'][i]) + ' ' + ' '.join(self.faces['F'][i]) + ' ' + ' '.join(self.faces['R'][i]) + ' ' + ' '.join(self.faces['B'][i]) + "\n")
            output.write("    " + print_face(self.faces['D']) + "\n")

    def random_move(self, output):
        move = random.choice(list(self.moves.keys()))
        self.moves[move]()
        output.write(f'Performed move: {move}\n')
        self.print_cube(output)

def main():
    parser = argparse.ArgumentParser(description="Simulate a Rubik's Cube and perform basic operations.")
    parser.add_argument('-s', '--shuffle', type=int, default=0, help="Number of random moves to shuffle the cube before starting to print")
    parser.add_argument('-m', '--moves', type=int, default=1, help="Number of moves to print to the stdout")
    parser.add_argument('-o', '--output', type=str, help="Optional output file to use instead of stdout")
    parser.add_argument('-c', '--condensed', action='store_true', help="Optional condensed form without spaces")
    args = parser.parse_args()

    if args.output:
        output = open(args.output, 'w')
    else:
        output = sys.stdout

    cube = RubiksCube(condensed_output=args.condensed)
    if args.shuffle > 0:
        cube.shuffle(args.shuffle)
        output.write(f"Shuffled cube with {args.shuffle} random moves:\n")
    else:
        output.write("Initial solved cube:\n")
    cube.print_cube(output)

    for _ in tqdm(range(args.moves), desc="Applying moves"):
        cube.random_move(output)

    if args.output:
        output.close()

if __name__ == "__main__":
    main()

