import argparse
import random
import sys
from tqdm import tqdm

class RubiksCube:
    def __init__(self, condensed_output=False, allowed_moves=None, shuffle_moves=None):
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
        all_moves = {
            'u': self.rotate_U,
            'U': self.rotate_U_inv,
            'e': self.rotate_E,
            'E': self.rotate_E_inv,
            'd': self.rotate_D,
            'D': self.rotate_D_inv,
            'f': self.rotate_F,
            'F': self.rotate_F_inv,
            's': self.rotate_S,
            'S': self.rotate_S_inv,
            'v': self.rotate_B,
            'V': self.rotate_B_inv,
            'l': self.rotate_L,
            'L': self.rotate_L_inv,
            'p': self.rotate_R,
            'P': self.rotate_R_inv,
            'm': self.rotate_M,
            'M': self.rotate_M_inv,
            'x': self.rotate_x,
            'X': self.rotate_x_inv,
            'i': self.rotate_y,
            'I': self.rotate_y_inv,
            'z': self.rotate_z,
            'Z': self.rotate_z_inv,
        }

        # Limit shuffle to the allowed move list if not None
        self.shuffle_moves = {k: v for k, v in all_moves.items() if shuffle_moves is None or k in shuffle_moves}

        # Limit to the allowed move list if not None
        self.moves = {k: v for k, v in all_moves.items() if allowed_moves is None or k in allowed_moves}

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

    def rotate_M(self):
        for i in range(3):
            self.faces['U'][i][1], self.faces['F'][i][1], self.faces['D'][i][1], self.faces['B'][2 - i][1] = \
                self.faces['B'][2 - i][1], self.faces['U'][i][1], self.faces['F'][i][1], self.faces['D'][i][1]

    def rotate_M_inv(self):
        for i in range(3):
            self.faces['U'][i][1], self.faces['B'][2 - i][1], self.faces['D'][i][1], self.faces['F'][i][1] = \
                self.faces['F'][i][1], self.faces['U'][i][1], self.faces['B'][2 - i][1], self.faces['D'][i][1]

    def rotate_E(self):
        for i in range(3):
            self.faces['F'][1][i], self.faces['L'][1][i], self.faces['B'][1][i], self.faces['R'][1][i] = \
                self.faces['R'][1][i], self.faces['F'][1][i], self.faces['L'][1][i], self.faces['B'][1][i]

    def rotate_E_inv(self):
        for i in range(3):
            self.faces['F'][1][i], self.faces['R'][1][i], self.faces['B'][1][i], self.faces['L'][1][i] = \
                self.faces['L'][1][i], self.faces['F'][1][i], self.faces['R'][1][i], self.faces['B'][1][i]

    def rotate_S(self):
        for i in range(3):
            self.faces['U'][1][i], self.faces['R'][i][1], self.faces['D'][1][2 - i], self.faces['L'][2 - i][1] = \
                self.faces['L'][2 - i][1], self.faces['U'][1][i], self.faces['R'][i][1], self.faces['D'][1][2 - i]

    def rotate_S_inv(self):
        for i in range(3):
            self.faces['U'][1][i], self.faces['L'][2 - i][1], self.faces['D'][1][2 - i], self.faces['R'][i][1] = \
                self.faces['R'][i][1], self.faces['U'][1][i], self.faces['L'][2 - i][1], self.faces['D'][1][2 - i]

    def rotate_x(self):
        self.rotate_R()
        self.rotate_M_inv()
        self.rotate_L_inv()

    def rotate_x_inv(self):
        self.rotate_R_inv()
        self.rotate_M()
        self.rotate_L()

    def rotate_y(self):
        self.rotate_U()
        self.rotate_E()
        self.rotate_D_inv()

    def rotate_y_inv(self):
        self.rotate_U_inv()
        self.rotate_E_inv()
        self.rotate_D()

    def rotate_z(self):
        self.rotate_F()
        self.rotate_S()
        self.rotate_B_inv()

    def rotate_z_inv(self):
        self.rotate_F_inv()
        self.rotate_S_inv()
        self.rotate_B()

    def shuffle(self, k):
        moves = list(self.shuffle_moves.keys())
        for _ in tqdm(range(k), desc="Shuffling"):
            move = random.choice(moves)
            self.shuffle_moves[move]()

    def print_cube(self, output):
        def print_face(face):
            return '\n'.join(''.join(row) for row in face)

        if self.condensed:
            output.write(print_face(self.faces['U']) + "\n")
            for i in range(3):
                output.write(''.join(self.faces['F'][i]) + ''.join(self.faces['R'][i]) + ''.join(self.faces['B'][i]) + ''.join(self.faces['L'][i]) + "\n")
            output.write(print_face(self.faces['D']) + "\n")
        else:
            output.write("   " + print_face(self.faces['U']).replace('\n', '\n   ') + "\n")
            for i in range(3):
                output.write(''.join(self.faces['L'][i]) + '' + ''.join(self.faces['F'][i]) + '' + ''.join(self.faces['R'][i]) + '' + ''.join(self.faces['B'][i]) + "\n")
            output.write("   " + print_face(self.faces['D']).replace('\n', '\n   ') + "\n")

    def random_move(self, output, prefix):
        move = random.choice(list(self.moves.keys()))
        self.moves[move]()
        output.write(f'{prefix}{move}\n')
        self.print_cube(output)

    def print_char_list(self, prefix, filename="char_list.txt"):
        with open(filename, 'w') as file:
            file.write(f"{prefix}\n")
            for face in self.faces:
                file.write(f"{face}\n")
            for move in self.moves:
                file.write(f"{move}\n")

    def interactive_mode(self):
        while True:
            self.print_cube(sys.stdout)
            move = input("Enter move (or 'q' to quit): ")
            if move == 'q':
                break
            if move in self.moves:
                self.moves[move]()
            else:
                print("Invalid move. Try again.")

def main():
    parser = argparse.ArgumentParser(description="Simulate a Rubik's Cube and perform basic operations.")
    parser.add_argument('-s', '--shuffle', type=int, default=0, help="Number of random moves to shuffle the cube before starting to print")
    parser.add_argument('-t', '--trials', type=int, default=1, help="Number of sets of shuffle moves + regular moves to do")
    parser.add_argument('-m', '--moves', type=int, default=1, help="Number of moves to print to the stdout")
    parser.add_argument('-o', '--output', type=str, help="Optional output file to use instead of stdout")
    parser.add_argument('-c', '--condensed', action='store_true', help="Optional condensed form without spaces")
    parser.add_argument('-p', '--prefix', type=str, default="@", help="Prefix to place before each move type, default is '@'")
    parser.add_argument('--charlist', action='store_true', help="Print the character list to char_list.txt")
    parser.add_argument('-i', '--interactive', action='store_true', help="Interactive mode to enter moves manually")
    parser.add_argument('-a', '--allowed_moves', type=str, default=None, nargs='*', help="List of allowed moves")
    parser.add_argument('-S', '--shuffle_moves', type=str, default=None, nargs='*', help="List of allowed moves for the shuffle stage")
    args = parser.parse_args()

    if args.output:
        output = open(args.output, 'w')
    else:
        output = sys.stdout

    cube = RubiksCube(condensed_output=args.condensed, allowed_moves=args.allowed_moves, shuffle_moves=args.shuffle_moves)
    if args.shuffle > 0:
        cube.shuffle(args.shuffle)
    cube.print_cube(output)

    if args.interactive:
        cube.interactive_mode()
    else:
        for _ in tqdm(range(args.moves), desc="Applying moves"):
            cube.random_move(output, args.prefix)

    if args.charlist:
        cube.print_char_list(args.prefix)

    if args.output:
        output.close()

if __name__ == "__main__":
    main()

