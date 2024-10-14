from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import *

# DDP
import os
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

# FSDP
from functools import partial
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Selective Checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper
)

# FP8 Transformer Engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from transformer_engine.pytorch.distributed import prepare_te_modules_for_fsdp


def train(
    cfg_path: str,
    bsz: int = 8,
    n_workers: int = 8,
    n_steps: int = 128*8,
    grad_acc_steps: int = 8,
    reduce_freq: int = 32,
    sac_freq: str = '1/1',
    use_fp8: bool = False,
    pt_compile: bool = False,
    compile_mode: str = 'default',
    profile: bool = False,
    output_dir: str = 'outputs/'
):
    '''
    :param       cfg_path: Model configuration file path
    :param            bsz: Batch size
    :param      n_workers: Number of CPUs for data loading
    :param        n_steps: Number of training steps
    :param grad_acc_steps: Number of gradient accumulation steps
    :param    reduce_freq: Number of steps FSDP performs an all gather
    :param       sac_freq: Selective activation checkpointing (AC). If sac_freq="q/p", applies AC for q out of every p blocks
    :param        use_fp8: Enable FP8
    :param     pt_compile: Enable PyTorch compile
    :param   compile_mode: Set PyTorch compile mode. Options: "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    :param        profile: Enable profiling
    :param     output_dir: Profiling output saving directory
    '''
    torch.manual_seed(3985)
    world_size = torch.cuda.device_count()
    train_args = (
        world_size,
        cfg_path, bsz, n_workers, n_steps, grad_acc_steps, reduce_freq,
        sac_freq, use_fp8, pt_compile, compile_mode, profile, output_dir
    )
    assert not (use_fp8 and (sac_freq != '1/1')), 'Selective AC currently doesn\'t work with Transformer Engine.'
    assert not (use_fp8 and pt_compile), 'PyTorch compile currently doesn\'t work with Transformer Engine.'

    try:
        mp.spawn(train_fsdp, train_args, nprocs=world_size)
    except:
        destroy_process_group()


def train_fsdp(
    rank, world_size,
    cfg_path, bsz, n_workers, n_steps, grad_acc_steps, reduce_freq,
    sac_freq, use_fp8, pt_compile, compile_mode, profile, output_dir
):
    # Construct process group
    os.environ.update({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '30985'})
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Configure training setup
    cfg_m, model_cls, blk_cls = get_model_config(cfg_path, use_fp8)
    model = model_cls(**asdict(cfg_m)).to(rank)
    dprint(rank, f'Loaded {model_cls} model.', end=' ')
    cfg_m.estimate_flops_per_token(model, bsz, rank)  # Need to do before wrapping in FSDP

    dataset = DummyDataset(cfg_m.vocab_size, cfg_m.max_seq_len, bsz*n_steps)
    data_loader = DataLoader(
        dataset, batch_size=bsz,
        num_workers=n_workers, pin_memory=True, shuffle=False,
        sampler=DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    )
    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f'{output_dir}/{Path(cfg_path).stem}_fsdp_trace.json'

    # FSDP
    all_gpus = dist.new_group(backend='nccl')
    model = FSDP(
        model,
        device_id=rank,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={blk_cls}),
        use_orig_params=True
    )
    dprint(rank, f'Created FSDP model')

    # Selective activation checkpointing
    block_idx = 0
    q, p = map(int, sac_freq.split('/'))

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
        dprint(rank, f'Configured selective activation checkpointing {sac_freq}')
    elif use_fp8:
        prepare_te_modules_for_fsdp(model)
        dprint(rank, 'Sharded TE modules for FSDP')

    # PyTorch compile
    if pt_compile:
        dprint(rank, f'Compiling in {compile_mode} mode')
        model = torch.compile(model, mode=compile_mode)

    # FP8
    fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo='max')

    # Training loop
    loop_iter = configure_train_loop(data_loader, profile, output_path, cfg_m, bsz, use_fp8, rank)
    ddp_loss = torch.zeros(2, device=rank)
    model.train()

    for step_idx, data_batch in loop_iter:
        input_BT, label_BT = map(lambda t: t.pin_memory().to(rank), data_batch)

        with torch.amp.autocast('cuda', torch.bfloat16):
            with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe, fp8_group=all_gpus):
                weight_cache = use_fp8 and (step_idx % grad_acc_steps == 0)
                logits_BTV = model(input_BT, is_first_microbatch=weight_cache)
                loss = F.cross_entropy(logits_BTV.flatten(0, 1), label_BT.flatten())
                loss /= grad_acc_steps

        loss.backward()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += input_BT.size(0)

        if (step_idx + 1) % grad_acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if (step_idx + 1) % reduce_freq == 0:
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    dist.barrier()
    destroy_process_group()


if __name__ == '__main__':
    import fire
    fire.Fire(train)
