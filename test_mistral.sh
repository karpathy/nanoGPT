#!/bin/bash

if [ -f "./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf" ]; then
  echo "./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf file found, continuing"
else
  echo "./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf file not found, downloading"
  wget -P ./models https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf
fi

python3 test.py

