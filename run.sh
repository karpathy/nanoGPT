#!/bin/bash
set -ax
python3 train.py \
	config/train_shakespeare_char.py \
	> /dev/null \
	2>&1 &
