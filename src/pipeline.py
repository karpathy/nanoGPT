import logging
import os
import subprocess

import json
import argparse

# Parse arg "config_file" from command line
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="./configs/config.json")
config_file = parser.parse_args().config_file

with open(config_file) as f:
    params = json.load(f)
    train_params = params['train_params']
    sample_params = params['sample_params']

train_cmd = "python train.py"
for key, value in train_params.items():
    train_cmd += f" --{key}={value}"

print(train_cmd)

sample_cmd = "python sample.py"
for key, value in sample_params.items():
    sample_cmd += f" --{key}={value}"

print(sample_cmd)
    
# Step 1: Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Step 2: Start training
logger.info("TRAINING MODEL")
print("TRAINING MODEL")
subprocess.run(train_cmd, shell=True, check=True)

# Step 3: Create sample
logger.info("CREATING SAMPLE")
print("CREATING SAMPLE")
subprocess.run(sample_cmd, shell=True, check=True)