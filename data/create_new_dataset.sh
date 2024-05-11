#!/bin/bash

new_dataset="${1}"
mkdir -p "${new_dataset}/utils"
pushd "$new_dataset"

# Use softlinks so we can use template/prepare.py for development
ln -s ../template/prepare.py prepare.py
ln -s ../template/utils/meta_util.py utils/meta_util.py
ln -s ../template/utils/txt_to_phonemes.sh utils/txt_to_phonemes.sh

# Different datasets may have different phoneme sets
cp ../template/utils/phoneme_list.txt utils/phoneme_list.txt

