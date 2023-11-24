#/bin/bash

# head to repo root
cd ../

dataset="shakespeare_char"
python3 "data/${dataset}/prepare.py"

softmax_variation=("constantmax" "polymax" "softermax" "sigsoftmax")

max_iters="3000"
block_size="256"
notes="check_all_softmax_variations"

# Loop over the array
for softmax_variant in "${softmax_variation[@]}"
do
  python3 train.py \
    --max_iters "$max_iters" \
    --eval_iters 200 \
    --eval_interval 200 \
    --log_interval 10 \
    --device cuda \
    --dataset "$dataset" \
    --use_softmax_variant \
    --softmax_variant "${softmax_variant}" \
    --use_softermax_xmax \
    --tensorboard_project "${dataset}_${softmax_variant}_${max_iters}" \
    --tensorboard_run_name "${softmax_variant}_${notes}" \
    --block_size "$block_size" \
    --out_dir "${dataset}_${softmax_variant}_${max_iters}_${notes}" \
    --compile
done

# do one iteration with regular softmax
softmax_variant="regular_softmax"
python3 train.py \
  --max_iters "$max_iters" \
  --eval_iters 200 \
  --eval_interval 200 \
  --log_interval 10 \
  --device cuda \
  --dataset "$dataset" \
  --no-use_softmax_variant \
  --use_softermax_xmax \
  --tensorboard_project "${dataset}_${softmax_variant}_${max_iters}" \
  --tensorboard_run_name "${softmax_variant}_${notes}" \
  --block_size "$block_size" \
  --out_dir "${dataset}_${softmax_variant}_${max_iters}_${notes}" \
  --compile

