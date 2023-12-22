#/bin/bash

# head to repo root
cd ../

dataset="shakespeare_char"
python3 "data/${dataset}/prepare.py"

device="cuda"
n_layer="2"
n_head="3"
n_embd="384"
max_iters="50"
block_size="32"
eval_iters="50"
eval_interval="50"
log_interval="10"
timestamp="$(date +%F_%T)"
notes="test_all_positional_embeddings"

pe_variant="rope"
run_name="${dataset}_${pe_variant}_${max_iters}_${block_size}_${n_layer}_${n_head}_${n_embd}_${notes}"
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
  --tensorboard_run_name "$run_name" \
  --out_dir "$output_dir"

pe_variant="rope_and_abs_pos_emb"
run_name="${dataset}_${pe_variant}_${max_iters}_${block_size}_${n_layer}_${n_head}_${n_embd}_${notes}"
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
  --use_abs_pos_embeddings \
  --use_rotary_embeddings \
  --rope_variant "rope" \
  --tensorboard_run_name "$run_name" \
  --out_dir "$output_dir"

pe_variant="abs_pos_emb"
run_name="${dataset}_${pe_variant}_${max_iters}_${block_size}_${n_layer}_${n_head}_${n_embd}_${notes}"
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
  --use_abs_pos_embeddings \
  --no-use_rotary_embeddings \
  --rope_variant "rope" \
  --tensorboard_run_name "$run_name" \
  --out_dir "$output_dir"

pe_variant="no_positional_embeddings"
run_name="${dataset}_${pe_variant}_${max_iters}_${block_size}_${n_layer}_${n_head}_${n_embd}_${notes}"
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
  --no-use_abs_pos_embeddings \
  --no-use_rotary_embeddings \
  --rope_variant "rope" \
  --tensorboard_run_name "$run_name" \
  --out_dir "$output_dir"

# Short Rope variations
for i in {2..16..2}; do
  pe_variant="shortrope_$i"
  run_name="${dataset}_${pe_variant}_${max_iters}_${block_size}_${n_layer}_${n_head}_${n_embd}_${notes}"
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
    --no-use_abs_pos_embeddings \
    --use_rotary_embeddings \
    --rope_variant "shortrope" \
    --shortrope_length "$i" \
    --tensorboard_run_name "$run_name" \
    --out_dir "$output_dir"

  pe_variant="abs_pos_emb_shortrope_$i"
  run_name="${dataset}_${pe_variant}_${max_iters}_${block_size}_${n_layer}_${n_head}_${n_embd}_${notes}"
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
    --use_abs_pos_embeddings \
    --use_rotary_embeddings \
    --rope_variant "shortrope" \
    --shortrope_length "$i" \
    --tensorboard_run_name "$run_name" \
    --out_dir "$output_dir"
done
