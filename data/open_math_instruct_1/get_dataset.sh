#!/bin/bash

wget -O trian.jsonl https://huggingface.co/datasets/nvidia/OpenMathInstruct-1/resolve/main/correct_solutions/train.jsonl?download=true
wget -O validation.jsonl https://huggingface.co/datasets/nvidia/OpenMathInstruct-1/resolve/main/correct_solutions/validation.jsonl?download=true

python3 jsonl_to_txt.py validation.jsonl validation.txt
python3 jsonl_to_txt.py train.jsonl train.txt

echo "Next step is to tokenize using the separate files."
echo "For example, with tiktoken:"
echo "python3 prepare.py -s -t train.txt -v validation.txt"

