import os
import random
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import hydra
from omegaconf import DictConfig, OmegaConf

from model import GPTConfig, GPT


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(s)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        cfg.data.gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_iter = cfg.data.gradient_accumulation_steps * ddp_world_size * cfg.data.batch_size * cfg.model.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(cfg.out_dir, exist_ok=True)

    set_seed(cfg.seed + seed_offset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    data_dir = os.path.join("data", cfg.data.dataset)

    def get_batch(split):
        if split == "train":
            data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        else:
            data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - cfg.model.block_size, (cfg.data.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+cfg.model.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+cfg.model.block_size]).astype(np.int64)) for i in ix])
        if device_type == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    iter_num = 0
    best_val_loss = 1e9

    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    model_args = dict(
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        block_size=cfg.model.block_size,
        bias=cfg.model.bias,
        vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
        dropout=cfg.model.dropout,
    )
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    optimizer = model.configure_optimizers(
        cfg.optim.weight_decay,
        cfg.optim.learning_rate,
        (cfg.optim.beta1, cfg.optim.beta2),
        device_type,
    )

    if cfg.compile:
        print("compiling the model...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(it):
        if it < cfg.optim.warmup_iters:
            return cfg.optim.learning_rate * (it + 1) / (cfg.optim.warmup_iters + 1)
        if it > cfg.optim.lr_decay_iters:
            return cfg.optim.min_lr
        decay_ratio = (it - cfg.optim.warmup_iters) / (cfg.optim.lr_decay_iters - cfg.optim.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.optim.min_lr + coeff * (cfg.optim.learning_rate - cfg.optim.min_lr)

    hydra_cfg_path = os.path.join(os.getcwd(), ".hydra", "config.yaml")

    if cfg.wandb_log and master_process:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            notes=f"hydra config: {hydra_cfg_path}",
        )

    raw_model = model.module if ddp else model
    X, Y = get_batch("train")
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    while True:
        lr = get_lr(iter_num) if cfg.optim.decay_lr else cfg.optim.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % cfg.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if cfg.wandb_log:
                import wandb
                wandb.log({"iter": iter_num, "train/loss": losses["train"], "val/loss": losses["val"], "lr": lr})
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    ckpt = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                            "model_args": model_args, "iter_num": iter_num, "best_val_loss": best_val_loss,
                            "config": OmegaConf.to_container(cfg)}
                    torch.save(ckpt, os.path.join(cfg.out_dir, "best.pt"))

        if iter_num == 0 and cfg.eval_only:
            break

        for micro_step in range(cfg.data.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == cfg.data.gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / cfg.data.gradient_accumulation_steps
            X, Y = get_batch("train")
            scaler.scale(loss).backward()

        if cfg.optim.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.log_interval == 0 and master_process:
            lossf = loss.item() * cfg.data.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(cfg.data.batch_size * cfg.data.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            if cfg.wandb_log and master_process:
                import wandb
                wandb.log({"iter": iter_num, "train/loss": lossf, "lr": lr, "mfu": running_mfu * 100})

        if iter_num > 0 and iter_num % cfg.checkpoint_interval == 0 and master_process:
            ckpt = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "model_args": model_args, "iter_num": iter_num, "best_val_loss": best_val_loss,
                    "config": OmegaConf.to_container(cfg)}
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_{iter_num:07d}.pt")
            print(f"saving periodic checkpoint to {ckpt_path}")
            torch.save(ckpt, ckpt_path)

        iter_num += 1
        local_iter_num += 1

        if iter_num > cfg.optim.max_iters:
            break

    if master_process:
        ckpt = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "model_args": model_args, "iter_num": iter_num, "best_val_loss": best_val_loss,
                "config": OmegaConf.to_container(cfg)}
        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pt"))
        print(f"saving last checkpoint to {cfg.out_dir}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
