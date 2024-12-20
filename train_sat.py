import os
import time
import math
import pickle
import argparse
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model_ext import GPTConfig, GPT
from sat_dataset import SATDataset, SATTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model on a given dataset.")

    # Required positional argument: config
    parser.add_argument('config', type=str, help='Path to the config file (.py) containing configuration settings.')

    # Original args
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=10)

    # nanoGPT args
    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--always_save_checkpoint', action='store_true')
    parser.add_argument('--init_from', type=str, default='scratch')
    parser.add_argument('--wandb_log', action='store_true')
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--rope', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--decay_lr', action='store_true')
    parser.add_argument('--warmup_iters', type=int, default=2000)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default=None)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--debug', action='store_true')

    # Arguments specific to SATDataset:
    parser.add_argument('--max_id', type=int, default=30, help='Maximum constant ID in the dataset')
    parser.add_argument('--shift_within_block', action='store_true')

    args = parser.parse_args()


    # Execute the config file
    config_vars = {}
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    with open(args.config, 'r') as f:
        code = f.read()
    exec(code, config_vars)

    # Update args from config file variables
    for k, v in config_vars.items():
        # Only update if k doesn't start with __ (to skip built-ins and such)
        if not k.startswith('__') and hasattr(args, k):
            print(f"Setting {k} to {v}")
            setattr(args, k, v)

    if args.dataset is None:
        raise ValueError("Please provide a dataset name in the config file or command line")
    
    if args.out_dir is None:
        args.out_dir = f"models/{args.dataset}-{time.strftime('%Y%m%d-%H%M%S')}"
    else:
        args.out_dir += f"-{time.strftime('%Y%m%d-%H%M%S')}"
    
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.dataset}-{time.strftime('%Y%m%d-%H%M%S')}"

    args.train_file = os.path.join(args.dataset, 'train.txt')

    return args

def main(args):
    if args.dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            args.dtype = 'bfloat16'
        else:
            args.dtype = 'float16'

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1337)

    tokenizer = SATTokenizer()

    train_dataset = SATDataset(
        file_path=args.train_file,
        tokenizer=tokenizer,
        block_size=args.block_size,
        max_id=args.max_id,
        split='train',         # Use most of the data for training
        val_ratio=0.01,        # 1% of data for validation
        remove_trace=False,
        shift_within_block=args.shift_within_block,
    )

    val_dataset = SATDataset(
        file_path=args.train_file,
        tokenizer=tokenizer,
        block_size=args.block_size,
        max_id=args.max_id,
        split='val',           # Use remaining 1% for validation
        val_ratio=0.01,
        remove_trace=False,
        shift_within_block=args.shift_within_block,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Attempt to derive vocab_size from tokenizer if needed
    vocab_size = len(tokenizer)
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=args.bias,
        vocab_size=vocab_size,
        dropout=args.dropout,
        use_rotary_emb=args.rope
    )

    if args.init_from == 'scratch':
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        iter_num = 0
        best_val_loss = 1e9
    elif args.init_from == 'resume':
        print(f"Resuming training from {args.out_dir}")
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        checkpoint = None

    model.to(args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)

    if args.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    @torch.no_grad()
    def estimate_loss(data_loader):
        model.eval()
        losses = []
        for i, batch in enumerate(data_loader):
            if i >= args.eval_iters:
                break
            with ctx:
                input_ids = batch["input_ids"].to(args.device)
                labels = batch["labels"].to(args.device)
                # attention_mask is optional depending on the model
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(args.device)

                logits, loss = model(input_ids, labels, attention_mask=attention_mask)
                losses.append(loss.item())
        model.train()
        return float(np.mean(losses)) if losses else float('inf')

    def get_lr(it, decay_iters):
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * it / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > decay_iters:
            return args.min_lr
        # 3) in between, use cosine decay
        decay_ratio = (it - args.warmup_iters) / (decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    if args.wandb_log:
        import wandb
        wandb.init(name=args.wandb_run_name, config=vars(args))

    # Calculate total training steps for progress bar
    total_steps = args.epochs * len(train_loader)

    # Create a tqdm progress bar
    pbar = tqdm(total=total_steps, desc="Training")

    t0 = time.time()
    local_iter_num = 0
    iter_num = 0
    best_val_loss = 1e9

    for epoch in range(args.epochs):
        for batch in train_loader:
            # Determine and set the learning rate for this iteration
            lr = get_lr(iter_num, total_steps) if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Periodic evaluation and checkpointing
            if iter_num % args.eval_interval == 0:
                train_loss = estimate_loss(train_loader)
                val_loss = estimate_loss(val_loader)
                print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
                if args.wandb_log:
                    wandb.log({"iter": iter_num, "train/loss": train_loss, "val/loss": val_loss, "lr": lr})
                if val_loss < best_val_loss or args.always_save_checkpoint:
                    best_val_loss = val_loss
                    if iter_num > 0 and not args.debug:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': vars(args),
                        }
                        print(f"saving checkpoint to {args.out_dir}")
                        torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))

            # If only evaluating, break out after the first iteration
            if iter_num == 0 and args.eval_only:
                break

            # Training step (no micro-steps)
            with ctx:
                input_ids = batch["input_ids"].to(args.device)
                labels = batch["labels"].to(args.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(args.device)

                logits, loss = model(input_ids, labels, attention_mask=attention_mask)

            scaler.scale(loss).backward()

            if args.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Update tqdm progress bar and print status
            pbar.update(1)
            pbar.set_postfix(iter=iter_num, loss=loss.item(), time_ms=f"{dt*1000:.2f}")

            iter_num += 1
            local_iter_num += 1

    # Close the progress bar after training finishes
    pbar.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
