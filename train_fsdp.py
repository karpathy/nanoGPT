import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from gpt import GPTConfig, GPT, GPTBlock
from llama import LLaMAConfig, LLaMA, LLaMABlock

import os
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

from functools import partial
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper
)


def train_fsdp(
    cfg_path: str,
    bsz: int = 8,
    n_workers: int = 8,
    n_steps: int = 128*8,
    grad_acc_steps: int = 8,
    reduce_freq: int = 32,
    sac_freq: str = '1/1',
    pt_compile: bool = False,
    compile_mode: str = 'default', # "default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"
    profile: bool = False,
    output_dir: str = 'outputs/fsdp/'
):
    torch.manual_seed(3985)
    world_size = torch.cuda.device_count()
    train_args = (
        world_size,
        cfg_path, bsz, n_workers, n_steps, grad_acc_steps, reduce_freq,
        sac_freq, pt_compile, compile_mode, profile, output_dir
    )
    mp.spawn(train, train_args, nprocs=world_size)


def train(
    rank, world_size,
    cfg_path, bsz, n_workers, n_steps, grad_acc_steps, reduce_freq,
    sac_freq, pt_compile, compile_mode, profile, output_dir
):
    # Construct process group
    os.environ.update({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '30985'})
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Configure model
    with open(cfg_path) as f:
        cfg_json = json.load(f)
    if cfg_json['arch_name'] == 'gpt':
        cfg_cls, model_cls, blk_cls = GPTConfig, GPT, GPTBlock
    elif cfg_json['arch_name'] == 'llama':
        cfg_cls, model_cls, blk_cls = LLaMAConfig, LLaMA, LLaMABlock
    else:
        raise ValueError(f'Model architecture {cfg_json["arch_name"]} not supported.')
    cfg_m = cfg_cls(**cfg_json)

    # Configure FSDP
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )
    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={blk_cls})
    
    model = model_cls(**cfg_json)
    # need to do before wrapping in FSDP
    flops_per_token = cfg_m.estimate_flops_per_token(model, cfg_json)
    model = FSDP(
        model,
        device_id=rank,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        use_orig_params=True
    )

    # Selective activation checkpointing
    # Reference: https://github.com/OrenLeung/fsdp/blob/main/fms_fsdp/policies/ac_handler.py
    block_idx = 0
    q, p = map(int, sac_freq.split('/'))  # Applies AC for q out of every p blocks
    def should_ckpt(submodule):
        nonlocal block_idx
        if isinstance(submodule, blk_cls):
            ckpt = (block_idx % p < q)
            block_idx += 1
            return ckpt
        return False
    if sac_freq != '1/1':
        non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=should_ckpt)

    if pt_compile:
        print("compiling")
        print(compile_mode)
        model = torch.compile(model, mode=compile_mode)

    # Configure training setup
    dataset = DummyDataset(cfg_m.vocab_size, cfg_m.max_seq_len, bsz*n_steps)
    data_loader = DataLoader(
        dataset, batch_size=bsz,
        num_workers=n_workers, pin_memory=True, shuffle=False,
        sampler=DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    )
    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    # Configure profiling
    if rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        flops_per_iter = flops_per_token * (bsz * cfg_m.max_seq_len)
        if 'H100' in torch.cuda.get_device_name():
            flops_promised = 989.5e12
        elif 'MI300X' in torch.cuda.get_device_name():
            flops_promised = 1300e12
        else:
            raise ValueError(f'FLOP/s for device {torch.cuda.get_device_name()} is unknown')

    if profile and rank == 0:
        def trace_export_callback(prof):
            prof.export_chrome_trace(f'{output_dir}/{Path(cfg_path).stem}_trace.json')

        prof_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
            with_flops=True,
            with_stack=True,
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir)
        )
    else:
        prof_ctx = nullcontext()

    # Training loop
    ddp_loss = torch.zeros(2, device=rank)
    pbar_ctx = tqdm(total=n_steps) if rank == 0 else nullcontext()
    model.train()

    with prof_ctx as prof, pbar_ctx as pbar:
        for step_idx, data_batch in enumerate(data_loader):
            if rank == 0:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            input_BT, label_BT = map(lambda t: t.pin_memory().to(rank, non_blocking=True), data_batch)

            with torch.amp.autocast('cuda', torch.bfloat16):
                logits_BTV = model(input_BT)
                loss = F.cross_entropy(logits_BTV.flatten(0, 1), label_BT.flatten())
                loss /= grad_acc_steps
            loss.backward()

            ddp_loss[0] += loss.item()
            ddp_loss[1] += input_BT.size(0)

            if (step_idx + 1) % grad_acc_steps == 0:  # Assume n_steps % grad_acc_steps == 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if (step_idx + 1) % reduce_freq == 0:  # Assume n_steps % reduce_freq == 0
                dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

            if rank == 0:
                end.record()
                torch.cuda.synchronize()

                t = start.elapsed_time(end) / 1e3
                flops_per_sec = flops_per_iter / t
                mfu = flops_per_sec / flops_promised

                pbar.set_description(f'[rank{rank}]  {(flops_per_sec/1e12):.2f} TFLOP/s  MFU={mfu:.2%}')

                pbar.update(world_size)
                
                if profile:
                    prof.step()

    dist.barrier()
    destroy_process_group()


class DummyDataset(Dataset):
    def __init__(self, vocab_size, max_seq_len, ds_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.ds_len = ds_len

    def __getitem__(self, idx):
        input_T = torch.randint(self.vocab_size, [self.max_seq_len], dtype=torch.int64)
        label_T = torch.cat([input_T[:-1], torch.randint(self.vocab_size, [1])])
        return input_T, label_T

    def __len__(self):
        return self.ds_len
 

if __name__ == '__main__':
    import fire
    fire.Fire(train_fsdp)
