#/bin/bash

# head to repo root
cd ../

python3 data/shakespeare_char/prepare.py

# Common settings
max_iterations=3000

# rope
python3 train.py \
  --max_iters "$max_iterations" \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "shakespeare_char" \
  --use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --rope_variant "rope" \
  --no-use_softmax_variant \
  --tensorboard_project "shkspr" \
  --tensorboard_run_name "rope" \
  --block_size 256 \
  --out_dir "shkspr_rope" \
  --compile

# abs pos
python3 train.py \
  --max_iters "$max_iterations" \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "shakespeare_char" \
  --no-use_rotary_embeddings \
  --use_abs_pos_embeddings \
  --no-use_softmax_variant \
  --tensorboard_project "shkspr" \
  --tensorboard_run_name "abs_pos" \
  --block_size 256 \
  --out_dir "shkspr_abs_pos" \
  --compile

# both abs pos and rope
python3 train.py \
  --max_iters "$max_iterations" \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "shakespeare_char" \
  --use_rotary_embeddings \
  --rope_variant "rope" \
  --use_abs_pos_embeddings \
  --no-use_softmax_variant \
  --tensorboard_project "shkspr" \
  --tensorboard_run_name "rope_abs_pos" \
  --block_size 256 \
  --out_dir "shkspr_rope_abs_pos" \
  --compile

no positional embeddings
python3 train.py \
  --max_iters "$max_iterations" \
  --eval_iters 200 \
  --eval_interval 100 \
  --log_interval 10 \
  --dataset "shakespeare_char" \
  --no-use_rotary_embeddings \
  --no-use_abs_pos_embeddings \
  --no-use_softmax_variant \
  --tensorboard_project "shkspr" \
  --tensorboard_run_name "no_pos_emb" \
  --block_size 256 \
  --out_dir "shkspr_nope" \
  --compile

# Short Rope variations
for i in {2..16..2}; do
  python3 train.py \
    --max_iters "$max_iterations" \
    --eval_iters 200 \
    --eval_interval 100 \
    --log_interval 10 \
    --dataset "shakespeare_char" \
    --use_rotary_embeddings \
    --rope_variant "shortrope" \
    --shortrope_length "${i}" \
    --no-use_abs_pos_embeddings \
    --no-use_softmax_variant \
    --tensorboard_project "shkspr" \
    --tensorboard_run_name "shortrope_${i}" \
    --block_size 256 \
    --out_dir "shkspr_rope_abs_pos_${i}" \
    --compile
done
