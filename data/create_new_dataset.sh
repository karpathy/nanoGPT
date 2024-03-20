#!/bin/bash

new_dataset="${1}"
mkdir "$new_dataset"
pushd "$new_dataset"

# Use softlinks so we can use template/prepare.py for development
ln -s ../template/prepare.py prepare.py
ln -s ../template/meta_util.py meta_util.py
ln -s ../template/txt_to_phonemes.sh txt_to_phonemes.sh

# Different datasets may have different phoneme sets
cp ../template/phoneme_list.txt .

