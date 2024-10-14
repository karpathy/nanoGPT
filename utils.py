import json
from contextlib import nullcontext

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from gpt import GPTConfig, GPT, GPTBlock, Fp8GPT, Fp8GPTBlock
from llama import LLaMAConfig, LLaMA, LLaMABlock, Fp8LLaMA, Fp8LLaMABlock


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


def get_model_config(cfg_path, fp8):
    with open(cfg_path) as f:
        cfg_json = json.load(f)

    if cfg_json['arch_name'] == 'gpt':
        cfg_cls = GPTConfig
        if fp8:
            model_cls, blk_cls = Fp8GPT, Fp8GPTBlock
        else:
            model_cls, blk_cls = GPT, GPTBlock
    elif cfg_json['arch_name'] == 'llama':
        cfg_cls = LLaMAConfig
        if fp8:
            model_cls, blk_cls = Fp8LLaMA, Fp8LLaMABlock
        else:
            model_cls, blk_cls = LLaMA, LLaMABlock
    else:
        raise ValueError(f'Model architecture {cfg_json["arch_name"]} not supported.')

    cfg_m = cfg_cls(**cfg_json)

    return cfg_m, model_cls, blk_cls


def configure_train_loop(data_loader, profile, output_path, cfg_m, bsz, fp8, rank=0):
    if rank != 0:
        for step_idx, data_batch in enumerate(data_loader):
            yield step_idx, data_batch
        return

    flops_per_iter = cfg_m.flops_per_token * (bsz * cfg_m.max_seq_len)

    if 'H100' in torch.cuda.get_device_name():
        flops_promised = 1979e12 if fp8 else 989.5e12
    elif 'MI300X' in torch.cuda.get_device_name():
        flops_promised = 2610e12 if fp8 else 1300e12
    else:
        raise ValueError(f'FLOP/s for device {torch.cuda.get_device_name()} is unknown')

    if profile:
        def trace_export_callback(prof):
            prof.export_chrome_trace(output_path)

        prof_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
            with_flops=True,
            on_trace_ready=trace_export_callback
        )
    else:
        prof_ctx = nullcontext()

    with prof_ctx as prof, tqdm(total=len(data_loader)) as pbar:
        for step_idx, data_batch in enumerate(data_loader):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            yield step_idx, data_batch

            end.record()
            torch.cuda.synchronize()

            t = start.elapsed_time(end) / 1e3
            flops_per_sec = flops_per_iter / t
            mfu = flops_per_sec / flops_promised

            pbar.set_description(f'[rank0]  {(flops_per_sec/1e12):.2f} TFLOP/s  MFU={mfu:.2%}')
            pbar.update()
            
            if profile:
                prof.step()


def dprint(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)
