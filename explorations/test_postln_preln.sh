#/bin/bash

# head to repo root
cd ../

dataset="shakespeare_char"
python3 "data/${dataset}/prepare.py"

ordering_options=("--use_post_ln" "--no-use_post_ln")

max_iters="3000"
block_size="256"
notes="test_post_ln_and_pre_ln"

# Loop over the array
for ordering_option in "${ordering_options[@]}"
do
  softmax_variant="regular_softmax"
  python3 train.py \
    --max_iters "$max_iters" \
    --eval_iters 200 \
    --eval_interval 200 \
    --log_interval 10 \
    --device cuda \
    --dataset "$dataset" \
    --no-use_softmax_variant \
    "${ordering_option}" \
    --use_softermax_xmax \
    --tensorboard_project "${dataset}_${softmax_variant}_${max_iters}" \
    --tensorboard_run_name "${softmax_variant}_${notes}_${ordering_option}" \
    --block_size "$block_size" \
    --out_dir "${dataset}_${softmax_variant}_${max_iters}_${notes}_${ordering_option}" \
    --compile
done

