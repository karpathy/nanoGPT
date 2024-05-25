#!/bin/bash

new_dataset="${1}"
mkdir -p "${new_dataset}"
pushd "$new_dataset"

# Use softlinks so we can use template/prepare.py for development
ln -s ../template/prepare.py prepare.py
ln -s ../template/utils ./utils

# Different datasets may have different phoneme sets
cp ../template/get_dataset.sh get_dataset.sh

