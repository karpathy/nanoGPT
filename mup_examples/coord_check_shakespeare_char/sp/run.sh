for width in 256 512 1024 2048 4096
do
    for seed in 1 2 3 4 5
    do
    head_size=64
    n_heads=$((width / head_size))
    out_dir="mup_examples/coord_check_shakespeare_char/sp/out/width${width}_depth2_seed${seed}"
    python train.py \
        --out_dir=$out_dir \
        --eval_interval=1 \
        --log_interval=1 \
        --eval_iters=1 \
        --eval_only=False \
        --always_save_checkpoint=False \
        --never_save_checkpoint=True \
        --init_from='scratch' \
        --wandb_log=False \
        --csv_log=True \
        --dataset='shakespeare_char' \
        --gradient_accumulation_steps=4 \
        --batch_size=2 \
        --block_size=1024 \
        --n_layer=2 \
        --n_head=$n_heads \
        --n_embd=$width \
        --dropout=0.0 \
        --bias=False \
        --init_std=0.02 \
        --learning_rate=1e-2 \
        --max_iters=10 \
        --weight_decay=1e-1 \
        --beta1=0.9 \
        --beta2=0.95 \
        --grad_clip=1.0 \
        --decay_lr=False \
        --seed=$seed \
        --backend='nccl' \
        --device='mps' \
        --dtype='float32' \
        --compile=False \
        --mup_enable_coord_check_logging=True
    done
done
