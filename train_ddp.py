import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from gpt import GPTConfig, GPT
from llama import LLaMAConfig, LLaMA

import os
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def train_ddp(
    cfg_path: str,
    bsz: int = 8,
    n_workers: int = 8,
    n_steps: int = 128*8,
    grad_acc_steps: int = 8,
    ckpt_freq: int = 64,
    pt_compile: bool = False,
    profile: bool = False,
    output_dir: str = 'outputs/ddp/'
):
    torch.manual_seed(3985)
    world_size = torch.cuda.device_count()
    train_args = (
        world_size, cfg_path, bsz, n_workers, n_steps, grad_acc_steps,
        ckpt_freq, pt_compile, profile, output_dir
    )
    mp.spawn(train, train_args, nprocs=world_size)


def train(
    rank, world_size,
    cfg_path, bsz, n_workers, n_steps, grad_acc_steps, ckpt_freq, pt_compile, profile, output_dir
):
    os.environ.update({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '30985'})
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

    with open(cfg_path) as f:
        cfg_json = json.load(f)
    match cfg_json['arch_name']:
        case 'gpt':
            cfg_cls, model_cls = GPTConfig, GPT
        case 'llama':
            cfg_cls, model_cls = LLaMAConfig, LLaMA
        case _:
            raise ValueError(f'Model architecture {cfg_json["arch_name"]} not supported.')
    cfg_m = cfg_cls(**cfg_json)
    model = DDP(model_cls(**cfg_json).to(rank))
    if pt_compile:
        model = torch.compile(model)

    dataset = SimulatedDataset(cfg_m.vocab_size, cfg_m.max_seq_len, bsz*n_steps)
    data_loader = DataLoader(
        dataset, batch_size=bsz,
        num_workers=n_workers, pin_memory=True, shuffle=False,
        sampler=DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    )
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)
    scaler = torch.amp.GradScaler()

    if rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if not profile:
            flops_per_token = cfg_m.estimate_flops_per_token(**cfg_json)
            flops_per_iter = 3 * flops_per_token * (bsz * cfg_m.max_seq_len)
    pbar_ctx = tqdm(total=n_steps) if rank == 0 else nullcontext()
    model.train()

    if profile and rank == 0:
        prof_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
            with_flops=True
        )
    else:
        prof_ctx = nullcontext()

    with prof_ctx as prof, pbar_ctx as pbar:
        for step_idx, data_batch in enumerate(data_loader):
            if rank == 0 and not profile:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            input_BT, label_BT = map(lambda t: t.pin_memory().to(rank, non_blocking=True), data_batch)

            with torch.amp.autocast('cuda', torch.float16):
                logits_BTV = model(input_BT)
                loss = F.cross_entropy(logits_BTV.flatten(0, 1), label_BT.flatten())
                loss /= grad_acc_steps
            scaler.scale(loss).backward()

            if (step_idx + 1) % grad_acc_steps == 0:  # Assume n_steps % grad_acc_steps == 0
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if rank == 0 and (step_idx + 1) % ckpt_freq == 0:  # Assume n_steps % ckpt_freq == 0
                ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    'scaler': scaler.state_dict()
                }
                torch.save(ckpt, Path(output_dir) / 'ckpt.pt')

            if rank == 0:
                if not profile:
                    end.record()
                    torch.cuda.synchronize()

                    t = start.elapsed_time(end) / 1e3
                    flops_per_sec = flops_per_iter / t
                    mfu = flops_per_sec / 989.5e12

                    pbar.set_description(f'[Rank {rank}]  {(flops_per_sec/1e12):.2f} TFLOP/s  MFU={mfu:.2%}')
                else:
                    prof.step()
                pbar.update(world_size)

    if rank == 0 and profile:
        prof.export_chrome_trace(f'{output_dir}/{Path(cfg_path).stem}_trace.json')

    destroy_process_group()


class SimulatedDataset(Dataset):
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
    fire.Fire(train_ddp)
