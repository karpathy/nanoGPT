#/bin/bash

# head to repo root
cd ../

dataset="shakespeare_char"
python3 "data/${dataset}/prepare.py"

ordering_options=("--use_post_ln" "--no-use_post_ln")

device="cuda"
n_layer="6"
n_head="6"
n_embd="384"
max_iters="5000"
eval_iters="200"
eval_interval="250"
log_interval="10"
block_size="256"

timestamp="$(date +%F_%T)"
notes="test_post_ln_and_pre_ln"

# Loop over the array
for ordering_option in "${ordering_options[@]}"
do
  run_name="${timestamp}_${ordering_option}_${notes}"
  output_dir="${timestamp}_${run_name}"
  python3 train.py \
    --max_iters "$max_iters" \
    --n_layer "$n_layer" \
    --n_head "$n_head" \
    --n_embd "$n_embd" \
    --block_size "$block_size" \
    --eval_iters "$eval_iters" \
    --eval_interval "$eval_interval" \
    --log_interval "$log_interval" \
    --device "$device" \
    --dataset "$dataset" \
    --use_rotary_embeddings \
    --rope_variant "rope" \
    "${ordering_option}" \
    --tensorboard_run_name "$run_name" \
    --out_dir "$output_dir"
done

