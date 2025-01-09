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
from transformers import AutoTokenizer, LlamaForCausalLM  # for HF llama

from model_ext import GPTConfig, GPT
from sat_dataset import SATDataset, SATTokenizer


# 1) HELPER FUNCTIONS
def create_custom_gpt(args, tokenizer):
    vocab_size = len(tokenizer)
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=args.bias,
        vocab_size=vocab_size,
        dropout=args.dropout,
        use_rotary_emb=args.rope,
        # [ADDED] pass your new config fields
        scale_attn_by_context=args.scale_attn_by_context,
        activation=args.activation,
        use_lstm_pos_enc=args.use_lstm_pos_enc,
    )

    if args.init_from == 'scratch':
        print("Initializing a new custom GPT model from scratch.")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        iter_num = 0
        best_val_loss = 1e9

    elif args.init_from == 'resume':
        print(f"Resuming custom GPT from {args.out_dir}")
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=args.device)

        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
                  # [ADDED] also handle new fields if present in the checkpoint
                  'scale_attn_by_context', 'activation', 'use_lstm_pos_enc'
                 ]:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                new_k = k[len(unwanted_prefix):]
                state_dict[new_k] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        checkpoint = None

    else:
        raise ValueError(f"Unknown init_from={args.init_from}")

    return model, iter_num, best_val_loss


def create_hf_llama(args, tokenizer):
    from transformers import LlamaConfig, LlamaForCausalLM
    vocab_size = len(tokenizer)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        max_position_embeddings=args.block_size,
    )

    if args.init_from == 'scratch':
        print("Initializing a new Hugging Face LLaMA model from scratch.")
        model = LlamaForCausalLM(config)
        iter_num = 0
        best_val_loss = 1e9
    elif args.init_from == 'resume':
        print(f"Resuming HF LLaMA from {args.out_dir}")
        try:
            model = LlamaForCausalLM.from_pretrained(args.out_dir, config=config)
            iter_num = 0
            best_val_loss = 1e9
        except Exception as e:
            raise ValueError(f"Failed to load HF LLaMA from {args.out_dir}: {e}")
    else:
        raise ValueError(f"Unknown init_from={args.init_from}")

    return model, iter_num, best_val_loss


def forward_model(model, input_ids, labels, attention_mask=None, huggingface_llama=False):
    if huggingface_llama:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        logits = outputs.logits
        loss = outputs.loss
        return logits, loss
    else:
        logits, loss = model(input_ids, labels, attention_mask=attention_mask)
        return logits, loss


# 2) MAIN TRAIN SCRIPT
def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT or HF LLaMA model on a given dataset.")

    parser.add_argument('config', type=str, help='Path to the config file (.py) containing configuration settings.')

    parser.add_argument('-hf', '--huggingface_llama', action='store_true')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--always_save_checkpoint', action='store_true')
    parser.add_argument('--init_from', type=str, default='scratch', choices=['scratch', 'resume'])
    parser.add_argument('--wandb_log', action='store_true')
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

    parser.add_argument('--max_id', type=int, default=30)
    parser.add_argument('--shift_within_block', action='store_true')

    # [ADDED] your old toggles
    parser.add_argument('--scale_attn_by_context', action='store_true',
                        help='Multiply each attention row by T*log(pos+1) in the non-flash path.')
    parser.add_argument('--activation', type=str, default='silu', choices=['silu','relu','gelu'],
                        help='Which MLP activation to use in custom GPT? (silu ~ LLaMA, or relu/gelu ~ older GPT2)')
    parser.add_argument('--use_lstm_pos_enc', action='store_true',
                        help='If set, run embeddings through an LSTM for positional info.')

    args = parser.parse_args()

    # Execute the config file
    config_vars = {}
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    with open(args.config, 'r') as f:
        code = f.read()
    exec(code, config_vars)

    for k, v in config_vars.items():
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

    # DATASET + TOKENIZER
    tokenizer = SATTokenizer()

    train_dataset = SATDataset(
        file_path=args.train_file,
        tokenizer=tokenizer,
        block_size=args.block_size,
        max_id=args.max_id,
        split='train',
        val_ratio=0.01,
        remove_trace=False,
        shift_within_block=args.shift_within_block,
    )
    val_dataset = SATDataset(
        file_path=args.train_file,
        tokenizer=tokenizer,
        block_size=args.block_size,
        max_id=args.max_id,
        split='val',
        val_ratio=0.01,
        remove_trace=False,
        shift_within_block=args.shift_within_block,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # MODEL CREATION
    if args.huggingface_llama:
        print("Using HuggingFace LLaMA model creation.")
        model, iter_num, best_val_loss = create_hf_llama(args, tokenizer)
        scaler = None  # HF LLaMA doesn't automatically use GradScaler in the same way
    else:
        print("Using custom GPT model creation.")
        model, iter_num, best_val_loss = create_custom_gpt(args, tokenizer)
        # We'll use GradScaler if float16
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    model.to(args.device)

    # Optional debug
    if args.debug and not args.huggingface_llama:
        print(model.summary())
        for name, param in model.named_parameters():
            print(name, param.shape, param.numel())

    # OPTIMIZER
    if args.huggingface_llama:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = model.configure_optimizers(
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            betas=(args.beta1, args.beta2),
            device_type=device_type
        )

    if args.compile:
        print("compiling the model... (PyTorch 2.x feature)")
        model = torch.compile(model)

    # EVALUATION
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
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(args.device)
                    attention_mask = attention_mask.to(args.device).bool()
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  


                logits, loss = forward_model(
                    model, input_ids, labels,
                    attention_mask=attention_mask,
                    huggingface_llama=args.huggingface_llama
                )
                losses.append(loss.item())
        model.train()
        return float(np.mean(losses)) if losses else float('inf')

    # LR SCHEDULING
    def get_lr(it, decay_iters):
        if it < args.warmup_iters:
            return args.learning_rate * it / args.warmup_iters
        if it > decay_iters:
            return args.min_lr
        decay_ratio = (it - args.warmup_iters) / (decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    # WANDB
    if args.wandb_log:
        import wandb
        wandb.init(name=args.wandb_run_name, config=vars(args))

    total_steps = args.epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc="Training")

    t0 = time.time()

    # TRAIN LOOP
    for epoch in range(args.epochs):
        for batch in train_loader:
            lr = get_lr(iter_num, total_steps) if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Evaluate checkpoint
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
                            'model_args': {} if args.huggingface_llama else {},
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': vars(args),
                        }
                        # [ADDED] if custom GPT, store the model_args so we can resume fully
                        if not args.huggingface_llama and hasattr(model, 'config'):
                            checkpoint['model_args'] = {
                                'n_layer': model.config.n_layer,
                                'n_head': model.config.n_head,
                                'n_embd': model.config.n_embd,
                                'block_size': model.config.block_size,
                                'bias': model.config.bias,
                                'vocab_size': model.config.vocab_size,
                                'scale_attn_by_context': model.config.scale_attn_by_context,  # new
                                'activation': model.config.activation,                      # new
                                'use_lstm_pos_enc': model.config.use_lstm_pos_enc,          # new
                            }

                        print(f"saving checkpoint to {args.out_dir}")
                        torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))

            if iter_num == 0 and args.eval_only:
                break

            # Forward pass
            with ctx:
                input_ids = batch["input_ids"].to(args.device)
                labels = batch["labels"].to(args.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(args.device)
                    attention_mask = attention_mask.bool()
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                logits, loss = forward_model(
                    model, input_ids, labels,
                    attention_mask=attention_mask,
                    huggingface_llama=args.huggingface_llama
                )

            # Backprop
            if not args.huggingface_llama:
                # custom GPT path with GradScaler
                scaler.scale(loss).backward()
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # HF approach
                loss.backward()
                if args.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            dt = time.time() - t0
            t0 = time.time()

            pbar.update(1)
            pbar.set_postfix(iter=iter_num, loss=loss.item(), time_ms=f"{dt*1000:.2f}")

            iter_num += 1

    pbar.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
