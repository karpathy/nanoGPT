"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run in debug mode example:
$ python train.py --batch_size=32 --other=args

To run DDP on 4 gpus on one node, example:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
import sys
import time
import math
from ast import literal_eval

from absl import app, flags
from ml_collections import config_flags
import wandb
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from config.train_default import get_config
from model import GPTConfig, GPT

_CONFIG = config_flags.DEFINE_config_file("config", "configs/train_default.py", "Training configuration.", lock_config=True)

def main(argv):
    print(_CONFIG)
    del argv
    cfg = flags.FLAGS.config
    
    # -----------------------------------------------------------------------------
    ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=cfg.backend)
        gpu_id = int(os.environ["LOCAL_RANK"])
        cfg.device = f"cuda:{gpu_id}"
    else:
        gpu_id = 0 # gpu_id 0 means this is the (single) master process, basically

    if gpu_id == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(1337 + gpu_id) # note: each worker gets a different seed
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # poor man's data loader, TODO evaluate need for actual DataLoader
    data_dir = os.path.join('data', cfg.dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+cfg.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+cfg.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(cfg.device), y.to(cfg.device)
        return x, y

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    model_args = dict(n_layer = cfg.n_layer, n_head = cfg.n_head, n_embd = cfg.n_embd, block_size = cfg.block_size, dropout = cfg.dropout)
    if cfg.init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif cfg.init_from == 'resume':
        print(f"Resuming training from {cfg.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(cfg.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        checkpoint_model_args = checkpoint['model_args']
        for k, v in model_args.items():
            assert checkpoint_model_args[k] == v, "for now"
            # TODO: think through how passed in params should interact with checkpoint params
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif cfg.init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {cfg.init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=cfg.dropout)
        model = GPT.from_pretrained(cfg.init_from, override_args)
        # read off and override the GPT sizing model args from the model config
        model_args['n_layer'] = model.config.n_layer
        model_args['n_head'] = model.config.n_head
        model_args['n_embd'] = model.config.n_embd
    # crop down the model block size if desired
    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)
    model.to(cfg.device)

    # optimizer
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, cfg.betas)
    if cfg.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])

    # compile the model
    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[gpu_id])

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = get_batch(split)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(iter, warmup_iters=cfg.warmup_iters, lr_decay_iters=cfg.lr_decay_iters, min_lr=cfg.min_lr):
        # 1) linear warmup for warmup_iters steps
        if iter < warmup_iters:
            return learning_rate * iter / warmup_iters
        # 2) if iter > lr_decay_iters, return min learning rate
        if iter > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # logging
    if cfg.wandb_log and gpu_id == 0:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.wandb_run_name, config=cfg.to_dict())

    # training loop
    t0 = time.time()
    while True:

        # determine the learning rate for this iteration
        if cfg.decay_lr:
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = cfg.learning_rate

        if iter_num % cfg.eval_interval == 0 and gpu_id == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if cfg.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                })
            if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                raw_model = model.module if ddp else model
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num == 0 and cfg.eval_only:
            break

        X, Y = get_batch('train')
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # TODO: gradient clipping evaluate need for
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0 and gpu_id == 0:
            lossf = loss.item() # loss as float. TODO CPU-GPU sync: profile, make sure not slow af
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1

        # termination conditions
        if iter_num > cfg.max_iters:
            break

    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    app.run(main)