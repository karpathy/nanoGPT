#!/usr/bin/env bash
set -euo pipefail

COMMON="wandb_log=true wandb_project=nanogpt wandb_group=week3-short seed=42 optim.max_iters=120 eval_interval=60 eval_iters=20 log_interval=20 checkpoint_interval=10000 compile=false"
OUT=/root/autodl-tmp/ckpt/week3
mkdir -p "$OUT" logs

run() {
  name="$1"
  shift
  echo "===== RUN $name ====="
  python train.py $COMMON wandb_run_name="$name" out_dir="$OUT/$name" "$@" 2>&1 | tee "logs/${name}.log"
}

# Day 2: parameter-level changes
run week3_day2_baseline
run week3_day2_lr3e-4 optim.learning_rate=0.0003
run week3_day2_lr1e-3 optim.learning_rate=0.001
run week3_day2_bs32 data.batch_size=32
run week3_day2_bs128 data.batch_size=128
run week3_day2_dropout0.1 model.dropout=0.1
run week3_day2_dropout0.2 model.dropout=0.2
run week3_day2_sgd optim.name=sgd optim.learning_rate=0.01 optim.weight_decay=0.0 optim.decay_lr=false

# Day 3: module-level changes
run week3_day3_silu model.activation=silu
run week3_day3_relu model.activation=relu
run week3_day3_head2 model.n_head=2
run week3_day3_head8 model.n_head=8
run week3_day3_postln model.norm_position=post

# Day 4: dense vs MoE changes
run week3_day4_dense
run week3_day4_moe_k2 model.use_moe=true model.moe_n_experts=4 model.moe_top_k=2 model.moe_aux_loss_weight=0.01
run week3_day4_moe_k1 model.use_moe=true model.moe_n_experts=4 model.moe_top_k=1 model.moe_aux_loss_weight=0.01
