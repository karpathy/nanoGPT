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


def train(
    cfg_path: str,
    gpu_id: int = 0,
    bsz: int = 8,
    n_workers: int = 8,
    n_steps: int = 128,
    grad_acc_steps: int = 8,
    pt_compile: bool = False,
    profile: bool = False,
    output_dir: str = 'outputs/single_gpu'
):
    torch.manual_seed(3985)
    torch.cuda.set_device(gpu_id)

    # Configure model
    with open(cfg_path) as f:
        cfg_json = json.load(f)
    if cfg_json['arch_name'] == 'gpt':
        cfg_cls, model_cls = GPTConfig, GPT
    elif cfg_json['arch_name'] == 'llama':
        cfg_cls, model_cls = LLaMAConfig, LLaMA
    else:
        raise ValueError(f'Model architecture {cfg_json["arch_name"]} not supported.')
    cfg_m = cfg_cls(**cfg_json)
    model = model_cls(**cfg_json).to(gpu_id)

    if pt_compile:
        model = torch.compile(model)

    # Configure training setup
    dataset = DummyDataset(cfg_m.vocab_size, cfg_m.max_seq_len, bsz*n_steps)
    data_loader = DataLoader(
        dataset, batch_size=bsz, num_workers=n_workers, pin_memory=True, shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    # Configure profiling
    if profile:
        prof_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5, repeat=1),
            with_flops=True
        )
    else:
        prof_ctx = nullcontext()
        flops_per_token = cfg_m.estimate_flops_per_token(**cfg_json)
        flops_per_iter = 3 * flops_per_token * (bsz * cfg_m.max_seq_len)
        if 'H100' in torch.cuda.get_device_name():
            flops_promised = 989.5e12
        elif 'MI300X' in torch.cuda.get_device_name():
            flops_promised = 1300e12
        else:
            raise ValueError(f'FLOP/s for device {torch.cuda.get_device_name()} is unknown')

    # Training loop
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.train()

    with prof_ctx as prof, tqdm(total=n_steps) as pbar:
        for step_idx, data_batch in enumerate(data_loader):
            if not profile:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            input_BT, label_BT = map(lambda t: t.pin_memory().to(gpu_id), data_batch)

            with torch.amp.autocast('cuda', torch.bfloat16):
                logits_BTV = model(input_BT)
                loss = F.cross_entropy(logits_BTV.flatten(0, 1), label_BT.flatten())
                loss /= grad_acc_steps
            loss.backward()

            if (step_idx + 1) % grad_acc_steps == 0:  # Assume n_steps % grad_acc_steps == 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not profile:
                end.record()
                torch.cuda.synchronize()

                t = start.elapsed_time(end) / 1e3
                flops_per_sec = flops_per_iter / t
                mfu = flops_per_sec / flops_promised

                pbar.set_description(f'{(flops_per_sec/1e12):.2f} TFLOP/s  MFU={mfu:.2%}')
            else:
                prof.step()
            pbar.update()

    if profile:
        prof.export_chrome_trace(f'{output_dir}/{Path(cfg_path).stem}_trace.json')


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
    fire.Fire(train)
