for width in 256 512 1024 2048
do
    for lr in 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625 0.0001220703125 0.00006103515625 0.00003051757812 0.00048828125 0.000244140625 0.0001220703125 0.00006103515625 0.00003051757812 0.00001525878906 0.000007629394531 0.000003814697266
    do
        for seed in 1 2 3
        do
            head_size=64
            n_heads=$((width / head_size))
            out_dir="mutransfer_lr/sp/out/width${width}_depth2_seed${seed}_lr${lr}"
            python train.py \
                --out_dir=$out_dir \
                --eval_interval=1 \
                --log_interval=1 \
                --eval_iters=1 \
                --eval_only=False \
                --skip_val_loss=True \
                --always_save_checkpoint=False \
                --never_save_checkpoint=True \
                --init_from='scratch' \
                --wandb_log=False \
                --csv_log=True \
                --dataset='shakespeare_char' \
                --gradient_accumulation_steps=8 \
                --batch_size=1 \
                --block_size=1024 \
                --n_layer=2 \
                --n_head=$n_heads \
                --n_embd=$width \
                --dropout=0.0 \
                --bias=False \
                --init_std=0.02 \
                --learning_rate=$lr \
                --max_iters=122 \
                --weight_decay=1e-1 \
                --beta1=0.9 \
                --beta2=0.95 \
                --grad_clip=1.0 \
                --decay_lr=False \
                --seed=$seed \
                --backend='nccl' \
                --device='mps' \
                --dtype='float32' \
                --compile=False
        done
    done
done
