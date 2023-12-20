#/bin/bash

# head to repo root
cd ../

dataset="shakespeare_char"
python3 "data/${dataset}/prepare.py"

# softmax_variation=("softmax" "constantmax" "polymax" "softermax" "sigsoftmax" "sigsoftmax_base2")
softmax_variation=("constantmax" "polymax" "softermax" "sigsoftmax")

n_layer="3"
n_head="3"
n_embd="384"
max_iters="1000"
block_size="64"
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
    --n_embd "$n_embd" \
    --eval_iters 200 \
    --eval_interval 200 \
    --log_interval 10 \
    --device cpu \
    --dataset "$dataset" \
    --softmax_variant_attn "${softmax_variant}" \
    --softmax_variant_output "softmax" \
    --softermax_use_xmax \
    --tensorboard_run_name "$run_name" \
    --block_size "$block_size" \
    --out_dir "${output_dir}"

  sleep 5
done
