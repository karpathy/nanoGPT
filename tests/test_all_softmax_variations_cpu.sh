#/bin/bash

# head to repo root
cd ../

dataset="shakespeare_char"
python3 "data/${dataset}/prepare.py"

softmax_variation=( \
  "softmax" \
  "consmax" \
  "consmax_quan" \
  "polymax" \
  "softermax" \
  "sigsoftmax" \
  "exppolymax" \
  "saturatingconsmax" \
  "strongermax")

n_layer="2"
n_head="2"
n_kv_group="2"
n_embd="60"
max_iters="50"
block_size="32"
eval_iters="50"
eval_interval="50"
timestamp="$(date +%F_%T)"
notes="check_all_softmax_variations"
run_name="${dataset}_${softmax_variant}_${max_iters}_${block_size}_${n_layer}_${n_head}_${n_embd}_${notes}"

# Loop over the array
for softmax_variant in "${softmax_variation[@]}"
do
  output_dir="results/${timestamp}_${notes}_${softmax_variant}"
  if [ ! -d "${output_dir}" ]; then
    mkdir -p "${output_dir}"
  fi

  python3 train.py \
    --max_iters "$max_iters" \
    --n_layer "$n_layer" \
    --n_head "$n_head" \
    --n_kv_group "$n_kv_group" \
    --n_embd "$n_embd" \
    --eval_iters "$eval_iters" \
    --eval_interval "$eval_interval" \
    --log_interval 10 \
    --device cpu \
    --dataset "$dataset" \
    --softmax_variant_attn "${softmax_variant}" \
    --softmax_variant_output "softmax" \
    --softermax_use_xmax \
    --tensorboard_run_name "$run_name" \
    --block_size "$block_size" \
    --out_dir "${output_dir}"

  python3 sample.py \
    --out_dir "${output_dir}" \
    --device "cpu" \
    --num_samples 1 \
    --max_new_tokens 100 \
    --start "What great fortune this is"

  sleep 3
done
