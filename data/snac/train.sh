#!/bin/bash

pushd ../../
python3 train.py --max_iters 10000 --decay_lr --device cuda --dataset snac --shared_fire_embeddings --no-use_abs_pos_embedding --use_fire_embeddings --compile --always_save_checkpoint
popd

