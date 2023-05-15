"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py -c batch_size=32 compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch
from pyhocon import ConfigTree
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from config.config_loader import to_dict, to_json, load_config
from model import GPTConfig, GPT


def train(conf: ConfigTree):
    # various inits, derived attributes, I/O setup
    print("config used =", to_json(conf))
    batch_size = conf.get_int("batch_size")
    block_size = conf.get_int("block_size")
    device = conf.get_string("device")
    gradient_accumulation_steps = conf.get_int("gradient_accumulation_steps")
    ddp = conf.get_int("ddp_rank") != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=conf.get("backend"))
        # torch.cuda.set_device(): Usage of this function is discouraged in favor of device.
        # In most cases it’s better to use CUDA_VISIBLE_DEVICES environmental variable.
        torch_device = torch.device(device)
        torch.cuda.device(torch_device)
        master_process = conf.get_int("ddp_rank") == 0  # this process will do logging, checkpointing etc.

        assert gradient_accumulation_steps % torch.cuda.device_count() == 0
        gradient_accumulation_steps //= torch.cuda.device_count()
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True

    seed_offset = conf.get_int("seed_offset")
    ddp_world_size = conf.get_int("ddp_world_size")
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    out_dir = conf.get_string("out_dir")
    dtype = conf.get_string("dtype")
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # poor man"s data loader
    train_data, val_data = load_data(conf)

    # init these up here, can override if init_from="resume" (i.e. from a checkpoint)

    # model init
    checkpoint = load_checkpoint(conf)
    iter_num = 0 if checkpoint is None else checkpoint["iter_num"]
    best_val_loss = 1e9 if checkpoint is None else checkpoint["best_val_loss"]
    model, model_args = get_model(checkpoint, conf)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # optimizer
    optimizer = get_optimizer(checkpoint, device_type, model, conf)
    checkpoint = None  # free up memory

    # compile the model
    if conf.get_bool("compile"):
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[conf.get_int("ddp_local_rank")])


    # logging
    wandb_log_enabled = tracking_init(master_process, conf)

    # training loop
    X, Y = get_batch("train", train_data, val_data, conf)  # fetch the very first batch

    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    eval_interval = conf.get_int("eval_interval")
    # helps estimate an arbitrarily accurate loss over either split using many batches

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, conf)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(ctx, model, train_data, val_data, conf)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log_enabled:
                wandb_log(losses, iter_num, lr, running_mfu)

            save_best_loss = False
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_best_loss = True

            save_checkpoint(save_best_loss, iter_num, optimizer, raw_model, model_args,best_val_loss, conf)

        if iter_num == 0 and conf.get_bool("eval_only"):
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train", train_data, val_data, conf)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # clip the gradient
        grad_clip = conf.get_float("grad_clip")
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        log_interval = conf.get_int("log_interval")
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        max_iters = conf.get_int("max_iters")
        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()


def get_batch(split, train_data1, val_data1, conf):
    device = conf.get_string("device")
    device_type = "cuda" if "cuda" in device else "cpu"
    batch_size = conf.get_int("batch_size")
    block_size = conf.get_int("block_size")

    data = train_data1 if split == "train" else val_data1
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(ctx, model, train_data, val_data, conf):
    eval_iters = conf.get_int("eval_iters")
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, conf)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, conf):
    learning_rate = conf.get_float("learning_rate")
    decay_lr1 = conf.get_bool("decay_lr")
    if not decay_lr1:
        return learning_rate

    warmup_iters = conf.get_int("warmup_iters")
    lr_decay_iters = conf.get_int("lr_decay_iters")
    min_lr = conf.get_float("min_lr")

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def save_checkpoint(save_best_loss,
                    iter_num,
                    optimizer,
                    raw_model,
                    model_args,
                    best_val_loss,
                    conf
                    ):
    out_dir = conf.get_string("out_dir")
    always_save_checkpoint = conf.get_bool("always_save_checkpoint")
    if save_best_loss or always_save_checkpoint:
        if iter_num > 0:
            checkpoint1 = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": to_dict(conf),
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint1, os.path.join(out_dir, "ckpt.pt"))


def load_data(train_conf):
    # poor man"s data loader
    dataset1 = train_conf.get_string("dataset")
    data_dir = os.path.join("data", dataset1)
    train_data1 = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data1 = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_data1, val_data1


def load_checkpoint(conf):
    init_from1 = conf.get_string("init_from")
    checkpoint1 = None
    if init_from1 == "resume":
        out_dir1 = conf.get_string("out_dir")
        device1 = conf.get_string("device")
        ckpt_path1 = os.path.join(out_dir1, "ckpt.pt")
        checkpoint1 = torch.load(ckpt_path1, map_location=device1)
    return checkpoint1


def get_model_from_checkpoint(checkpoint):
    checkpoint_model_args1 = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can"t even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    model_args = {}
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args1[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model, model_args


def get_vocab_size(conf):
    # attempt to derive vocab_size from the dataset
    dataset = conf.get_string("dataset")
    data_dir = os.path.join("data", dataset)
    meta_path = os.path.join(data_dir, "meta.pkl")
    vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta1 = pickle.load(f)
        vocab_size = meta1["vocab_size"]
        print(f"found vocab_size = {vocab_size} (inside {meta_path})")

    if vocab_size is None:
        vocab_size = conf.get_int("vocab_size")
        vocab_size_reason = conf.get_string("vocab_size_reason")
        print(vocab_size_reason)

    return vocab_size


def get_model(checkpoint, conf):
    block_size = conf.get_int("block_size")
    n_layer = conf.get_int("n_layer")
    n_head = conf.get_int("n_head")
    n_embd = conf.get_int("n_embd")
    bias = conf.get_bool("bias")
    dropout = conf.get_float("dropout")
    out_dir = conf.get_string("out_dir")
    device = conf.get_string("device")
    meta_vocab_size = get_vocab_size(conf)

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                       bias=bias, vocab_size=None, dropout=dropout)  # start with model_args from command line
    init_from = conf.get_string("init_from")
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we"ll use for from-scratch training
        model_args["vocab_size"] = meta_vocab_size
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        model, model_args = get_model_from_checkpoint(checkpoint)

    elif init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args1 = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args1)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args["block_size"] = block_size  # so that the checkpoint will have the right value
    model.to(device)

    return model, model_args


def get_optimizer(checkpoint, device_type, model, conf):
    init_from = conf.get_string("init_from")
    weight_decay1 = conf.get_float("weight_decay")
    learning_rate1 = conf.get_float("learning_rate")
    beta1 = conf.get_float("beta1")
    beta2 = conf.get_float("beta2")
    optimizer = model.configure_optimizers(weight_decay1, learning_rate1, (beta1, beta2), device_type)
    if init_from == "resume" and checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return optimizer


def tracking_init(master_process, conf):
    # wandb logging
    wandb_log_enabled1 = "wandb_log" in conf and conf.get_bool("wandb_log")
    if master_process and wandb_log_enabled1:
        wandb_init(conf)
    return wandb_log_enabled1


def wandb_init(conf):
    import wandb
    project = conf.get_string("wandb_project")
    run_name = conf.get_string("wandb_run_name")
    run_name = f"{run_name}-{str(time.time())}"
    wandb.init(project=project, name=run_name, config=to_dict(conf))


def wandb_log(losses, iter_num, lr, running_mfu):
    import wandb
    wandb.log({
        "iter": iter_num,
        "train/loss": losses["train"],
        "val/loss": losses["val"],
        "lr": lr,
        "mfu": running_mfu * 100,  # convert to percentage
    })


def main():
    conf = load_config()
    assert len(conf) > 0, "no configuration provided"
    train(conf)


if __name__ == "__main__":
    main()
