#/bin/bash

cd ../
python3 train.py \
  --max_iters 3000 \
  --dataset csv_data \
  --tensorboard_project csv_data \
  --tensorboard_run_name csv_data

