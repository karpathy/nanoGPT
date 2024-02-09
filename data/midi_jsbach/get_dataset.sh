#!/bih/bash

git clone https://github.com/czhuang/JSB-Chorales-dataset.git

cp JSB-Chorales-dataset/Jsb16thSeparated.json midi.json

python3 convert_json_to_csv.py
python3 convert_base_to_base.py midi.csv midi_12.csv --input_base 10 --output_base 12


