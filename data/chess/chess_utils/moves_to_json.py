import json
import os

def parse_chess_data(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    games = []
    game_data = {}

    for line in lines:
        if line.startswith('[Event'):
            # If there's an ongoing game data, save it before starting a new one
            if game_data:
                games.append(game_data)
                game_data = {}
            game_data['Event'] = line.strip().split('"')[1]
        elif line.startswith('[Result'):
            game_data['Result'] = line.strip().split('"')[1]
        elif line.startswith('[Termination'):
            game_data['Termination'] = line.strip().split('"')[1]
        elif line.startswith('[WhiteElo'):
            game_data['WhiteElo'] = line.strip().split('"')[1]
        elif line.startswith('[BlackElo'):
            game_data['BlackElo'] = line.strip().split('"')[1]
        elif line.strip().startswith('1.'):
            game_data['Moveset'] = line.strip()

    # Append the last game if not already added
    if game_data:
        games.append(game_data)

    # Write to output file as JSON
    with open(output_file_path, 'w') as outfile:
        for game in games:
            json.dump(game, outfile)
            outfile.write('\n')

# Specify the path to your dataset and the output file
input_file_path = 'datasets/lichess_games.txt'
json_dir = 'json/'
output_filename = 'parsed_games.json'
output_path = os.path.join(json_dir, output_filename)

# Ensure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Check if the file is already downloaded and decompressed
if os.path.exists(output_path):
    print(f"{output_path} already exists. Skipping download.")

# Call the function to parse and write the data
parse_chess_data(input_file_path, output_path)

