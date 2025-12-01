"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""
# Changes made here will be reflected in train.py
# Still allows overriding config.yaml with --key=value

import yaml
import argparse
import os

def update_config(config, key, value):
    parts = key.split(".")
    for part in parts[:-1]:
        config = config.setdefault(part, {})
    config[parts[-1]] = value

def load_config(env):
    # Load the YAML configuration
    with open('config.yaml', 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Get the configuration for the specified environment
    if env not in yaml_config:
        raise ValueError(f"Environment '{env}' not found in the configuration.")
    config = yaml_config[env]

    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # Add arguments for each configuration key
    for key in config.keys():
        if key == 'learning_rate':
            parser.add_argument(f'--{key}', type=float, default=argparse.SUPPRESS)
        elif isinstance(config[key], dict):
            for sub_key in config[key].keys():
                parser.add_argument(f'--{key}.{sub_key}', type=type(config[key][sub_key]), default=argparse.SUPPRESS)
        else:
            parser.add_argument(f'--{key}', type=type(config[key]), default=argparse.SUPPRESS)

    args = parser.parse_args()

    # Update the configuration with command-line arguments
    for key, value in vars(args).items():
        update_config(config, key, value)

    return config

# Get the environment from the command line or default to 'default'
env = os.environ.get('ENV', 'default')

# Load the configuration for the specified environment
config = load_config(env)