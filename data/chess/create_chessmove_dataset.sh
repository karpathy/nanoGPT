#!/bin/bash

python3 get_dataset.py
python3 process_games.py
python3 moves_to_json.py
python3 filter.py
python3 extract_moveset.py
python3 create_moveset_input.py

