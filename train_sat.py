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

# If you have your own LlamaConfig definition:
# from your_llama_config_file import LlamaConfig
# or use the built-in one from Hugging Face (transformers>=4.28)

from model_ext import GPTConfig, GPT
from sat_dataset import SATDataset, SATTokenizer

# ----------------------------------------------------------------------
# 1) HELPER FUNCTIONS
# ----------------------------------------------------------------------

def create_custom_gpt(args, tokenizer):
    """
    Build or resume a custom GPT model (nanoGPT style) based on `args`.
    Returns model, iter_num, best_val_loss.
    """
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
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
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

    return model, 0, 1e9 if args.init_from == 'scratch' else (model, iter_num, best_val_loss)


def create_hf_llama(args, tokenizer):
    """
    Build or resume a Hugging Face LLaMA model.
    Returns model, iter_num, best_val_loss.

    If you'd like to replicate the architecture exactly, set fields in LlamaConfig
    to match your desired hyperparameters. This example does minimal overrides.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    vocab_size = len(tokenizer)

    # Construct a LlamaConfig from your script arguments
    # (You can expand this if you want to match the entire LLaMA architecture.)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        max_position_embeddings=args.block_size,
        # Possibly set:
        # intermediate_size=4 * args.n_embd,  # typical LLaMA uses ~4x hidden
        # tie_word_embeddings=False,          # If you want the embedding + lm_head untied
        # attention_dropout=args.dropout,
        # attention_bias=not args.bias
    )

    if args.init_from == 'scratch':
        print("Initializing a new Hugging Face LLaMA model from scratch.")
        model = LlamaForCausalLM(config)
        iter_num = 0
        best_val_loss = 1e9

    elif args.init_from == 'resume':
        print(f"Resuming HF LLaMA from {args.out_dir}")
        # If you have a local HF checkpoint folder with config.json + pytorch_model.bin
        # you can do:
        # model = LlamaForCausalLM.from_pretrained(args.out_dir, config=config)
        # If your out_dir only has a single 'ckpt.pt' (nanoGPT style), you'd need a custom load.
        # This example tries the standard HF approach:
        try:
            model = LlamaForCausalLM.from_pretrained(args.out_dir, config=config)
            # If you had stored iteration info separately, load it here. For now, default:
            iter_num = 0
            best_val_loss = 1e9
        except Exception as e:
            raise ValueError(f"Failed to load HF LLaMA from {args.out_dir}: {e}")

    else:
        raise ValueError(f"Unknown init_from={args.init_from}")

    return model, 0, 1e9 if args.init_from == 'scratch' else (model, iter_num, best_val_loss)


def forward_model(model, input_ids, labels, attention_mask=None, huggingface_llama=False):
    """
    A small helper to unify forward pass across custom GPT & huggingface LLaMA.
    Returns (logits, loss).
    """
    if huggingface_llama:
        # HF LLaMA returns a CausalLMOutputWithPast, with .logits and .loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        logits = outputs.logits  # shape (batch, seq, vocab_size)
        loss = outputs.loss      # scalar
        return logits, loss

    else:
        # Custom GPT returns (logits, loss) directly
        logits, loss = model(input_ids, labels, attention_mask=attention_mask)
        return logits, loss


# ----------------------------------------------------------------------
# 2) MAIN TRAIN SCRIPT
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT or HF LLaMA model on a given dataset.")

    # Required: config
    parser.add_argument('config', type=str,
                        help='Path to the config file (.py) containing configuration settings.')

    # Additional flags
    parser.add_argument('-hf', '--huggingface_llama', action='store_true',
                        help='If set, build a Hugging Face LLaMA model instead of a custom GPT model.')

    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=10)

    # nanoGPT-like arguments
    parser.add_argument('--eval_interval', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--eval_iters', type=int, default=200)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--always_save_checkpoint', action='store_true')
    parser.add_argument('--init_from', type=str, default='scratch',
                        choices=['scratch', 'resume'])

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

    # SATDataset args
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
    # Decide dtype
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

    # ------------------------------------------------------------------
    # DATASET + TOKENIZER
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # MODEL CREATION
    # ------------------------------------------------------------------
    if args.huggingface_llama:
        print("Using HuggingFace LLaMA model creation.")
        model, iter_num, best_val_loss = create_hf_llama(args, tokenizer)
    else:
        print("Using custom GPT model creation.")
        model, iter_num, best_val_loss = create_custom_gpt(args, tokenizer)

    model.to(args.device)

    # Debug info
    if args.debug and not args.huggingface_llama:
        # custom GPT might have a .summary() method
        print(model.summary())
        for name, param in model.named_parameters():
            print(name, param.shape, param.numel())

    # ------------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------------
    if args.huggingface_llama:
        # For HF models, you might need your own config or a standard AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        # custom GPT has a .configure_optimizers(...) method
        scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
        optimizer = model.configure_optimizers(
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            betas=(args.beta1, args.beta2),
            device_type=device_type
        )

    if args.compile:
        print("compiling the model... (PyTorch 2.x feature)")
        model = torch.compile(model)

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
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

                logits, loss = forward_model(
                    model, input_ids, labels,
                    attention_mask=attention_mask,
                    huggingface_llama=args.huggingface_llama
                )
                losses.append(loss.item())
        model.train()
        return float(np.mean(losses)) if losses else float('inf')

    # ------------------------------------------------------------------
    # LR SCHEDULING
    # ------------------------------------------------------------------
    def get_lr(it, decay_iters):
        # linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * it / args.warmup_iters
        # if it > decay_iters, return min learning rate
        if it > decay_iters:
            return args.min_lr
        # in between, use cosine decay
        decay_ratio = (it - args.warmup_iters) / (decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    # ------------------------------------------------------------------
    # WANDB (optional)
    # ------------------------------------------------------------------
    if args.wandb_log:
        import wandb
        wandb.init(name=args.wandb_run_name, config=vars(args))

    total_steps = args.epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc="Training")

    t0 = time.time()
    local_iter_num = 0

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    for epoch in range(args.epochs):
        for batch in train_loader:
            lr = get_lr(iter_num, total_steps) if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Periodic eval + checkpoint
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
                            # if huggingface_llama, we may not have a model_args
                            'model_args': {} if args.huggingface_llama else {}, 
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': vars(args),
                        }
                        # If custom GPT, store model_args to allow resuming
                        if not args.huggingface_llama and hasattr(model, 'config'):
                            checkpoint['model_args'] = {
                                'n_layer': model.config.n_layer,
                                'n_head': model.config.n_head,
                                'n_embd': model.config.n_embd,
                                'block_size': model.config.block_size,
                                'bias': model.config.bias,
                                'vocab_size': model.config.vocab_size,
                            }

                        print(f"saving checkpoint to {args.out_dir}")
                        torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))

            # If eval-only, break after first iteration
            if iter_num == 0 and args.eval_only:
                break

            # Forward + Backprop
            with ctx:
                input_ids = batch["input_ids"].to(args.device)
                labels = batch["labels"].to(args.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(args.device)

                logits, loss = forward_model(
                    model, input_ids, labels,
                    attention_mask=attention_mask,
                    huggingface_llama=args.huggingface_llama
                )

            # If custom GPT uses GradScaler, do:
            if not args.huggingface_llama:
                # custom GPT approach
                scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
                scaler.scale(loss).backward()
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # HF approach, just normal backprop
                loss.backward()
                if args.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            pbar.update(1)
            pbar.set_postfix(iter=iter_num, loss=loss.item(), time_ms=f"{dt*1000:.2f}")

            iter_num += 1
            local_iter_num += 1

    pbar.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
