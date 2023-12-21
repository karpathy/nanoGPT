#!/bin/bash

# Check if the nano_gpt environment exists
conda env list | grep nano_gpt &> /dev/null
if [ $? -eq 0 ]; then
    echo "Activating nano_gpt environment"
    conda activate nano_gpt
else
    echo "nano_gpt environment does not exist. Let's create it..."
    conda env create -f environment.yml
    conda activate nano_gpt
fi