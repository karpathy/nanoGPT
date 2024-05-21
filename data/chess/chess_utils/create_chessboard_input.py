import chess
import argparse
from tqdm import tqdm

def print_compact_ascii_board(board, file=None):
    output = []
    for rank in range(7, -1, -1):
        row = ""
        for file_idx in range(8):
            piece = board.piece_at(chess.square(file_idx, rank))
            symbol = piece.symbol() if piece else "0"
            row += symbol
        output.append(row)
    if file:
        file.write("\n".join(output))
    else:
        print("\n".join(output))

def resolve_ambiguous_move(board, move):
    if len(move) == 4 and move[0] in "NBRQK" and move[1] in "abcdefgh":
        piece_type = {'N': chess.KNIGHT, 'B': chess.BISHOP, 'R': chess.ROOK, 'Q': chess.QUEEN, 'K': chess.KING}[move[0]]
        start_file = 'abcdefgh'.index(move[1])
        end_square = chess.SQUARE_NAMES.index(move[2:])
        candidates = []
        for i in range(8):
            square = chess.square(start_file, i)
            if board.piece_at(square) and board.piece_at(square).piece_type == piece_type and board.piece_at(square).color == board.turn:
                if chess.Move(square, end_square) in board.legal_moves:
                    candidates.append(chess.Move(square, end_square))
        if len(candidates) == 1:
            return candidates[0]
    return None

def apply_moves_and_print_boards(moves, output_file=None):
    board = chess.Board()
    if output_file:
        output_file.write(f"\n_\n")
        print_compact_ascii_board(board, output_file)
    else:
        print_compact_ascii_board(board)
    last_move = "~"  # Initialize last_move as "~" for the start
    for move in moves.split():
        try:
            board.push_san(move)
            if last_move == "~" and board.turn == chess.BLACK:  # Special case for the first move
                current_move = f"~W{move}"
            else:
                # ~W for response move or ~D for response move, except for first move.
                # Can anchor on these for interactive mode
                current_move = f"D{last_move}\n~W{move}" if board.turn == chess.BLACK else f"W{last_move}\n~D{move}"
            if output_file:
                output_file.write(f"\n{current_move}\n")
                print_compact_ascii_board(board, output_file)
            else:
                print(f"\n{current_move}\n")
                print_compact_ascii_board(board)
            last_move = move
        except ValueError:
            resolved_move = resolve_ambiguous_move(board, move)
            if resolved_move:
                board.push(resolved_move)
                if output_file:
                    output_file.write(f"{move} (resolved)\n")
                    print_compact_ascii_board(board, output_file)
                else:
                    print(f"{move} (resolved)\n")
                    print_compact_ascii_board(board)
            else:
                if output_file:
                    output_file.write(f"Skipping invalid or unresolved move: {move}\n")
                else:
                    print(f"Skipping invalid or unresolved move: {move}")
                break

def main():
    parser = argparse.ArgumentParser(description="Process a file of chess games and print each board state in ASCII format.")
    parser.add_argument('-i', "--filename", default='movesets_txt/moveset.txt', type=str, help="The filename containing the chess games, each on a new line.")
    parser.add_argument('-o', "--output", default='input.txt', type=str, help="Optional filename to write output to a text file.")
    args = parser.parse_args()

    if args.output:
        with open(args.output, 'w') as output_file:
            process_games_from_file(args.filename, output_file)
    else:
        process_games_from_file(args.filename)

def process_games_from_file(filename, output_file=None):
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Processing games"):
            if line.strip():  # Ensure the line is not empty
                moves = line.strip()
                apply_moves_and_print_boards(moves, output_file)

if __name__ == "__main__":
    main()

