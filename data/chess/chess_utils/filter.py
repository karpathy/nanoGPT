import json
import argparse
import os

def is_valid_elo(elo):
    try:
        return int(elo)
    except ValueError:
        return False

def filter_games(input_file_path, output_file_path, min_elo):
    with open(input_file_path, 'r') as infile:
        games = [json.loads(line) for line in infile]

    filtered_games = []
    for game in games:
        white_elo = is_valid_elo(game['WhiteElo'])
        black_elo = is_valid_elo(game['BlackElo'])
        if game['Termination'] == 'Normal' and white_elo and black_elo and white_elo >= min_elo and black_elo >= min_elo:
            filtered_games.append(game)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w') as outfile:
        json.dump(filtered_games, outfile, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Filter chess games based on termination type and minimum Elo ratings.")
    parser.add_argument('-i', '--input_file', type=str, default='json/parsed_games.json', help="Path to the input JSON file containing the games.")
    parser.add_argument('-o', '--output_file', type=str, default='filtered_json/filtered_games.json', help="Path to the output JSON file to store filtered games.")
    parser.add_argument('--min_elo', type=int, default=1500, help="Minimum Elo rating for both players. Default is 1500.")

    args = parser.parse_args()


    filter_games(args.input_file, args.output_file, args.min_elo)

if __name__ == "__main__":
    main()

