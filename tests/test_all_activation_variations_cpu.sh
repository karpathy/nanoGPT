#/bin/bash

# head to repo root
cd ../

dataset="shakespeare_char"
python3 "data/${dataset}/prepare.py"

activation_variation=("celu"
            "elu"
            "gelu"
            # "glu"
            "leaky_relu"
            "mish"
            "prelu"
            "relu6"
            "rrelu"
            "selu"
            "sigmoid"
            "silu"
            "softplus"
            "softsign"
            "squared_relu"
            "tanh")

n_layer="2"
n_head="3"
n_embd="384"
max_iters="50"
block_size="32"
eval_iters="50"
eval_interval="50"
timestamp="$(date +%F_%T)"
notes="check_all_activation_variations"
run_name="${dataset}_${activation_variant}_${max_iters}_${block_size}_${n_layer}_${n_head}_${n_embd}_${notes}"

# Loop over the array
for activation_variant in "${activation_variation[@]}"
do
  output_dir="results/${timestamp}_${notes}_${activation_variant}"
  if [ ! -d "${output_dir}" ]; then
    mkdir -p "${output_dir}"
  fi

  python3 train.py \
    --max_iters "$max_iters" \
    --n_layer "$n_layer" \
    --n_head "$n_head" \
    --n_embd "$n_embd" \
    --eval_iters "$eval_iters" \
    --eval_interval "$eval_interval" \
    --log_interval 10 \
    --device cpu \
    --dataset "$dataset" \
    --activation_variant "${activation_variant}" \
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
