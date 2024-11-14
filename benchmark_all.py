import json
import time
from pathlib import Path

import torch

from train import train as train_single
from train_ddp import train as train_ddp
from train_fsdp import train as train_fsdp


H100_CONFIG = [
    {
    "cfg_path": "configs/gpt2-1.5b.json",
    "strategy": "DDP",
    "dtype": "BF16",
    "device_name": "H100",
    "bsz": 12
    },
    {
    "cfg_path": "configs/gpt2-1.5b.json",
    "strategy": "DDP",
    "dtype": "FP8",
    "device_name": "H100",
    "bsz": 14
    },
    {
    "cfg_path": "configs/llama-3.1-8b.json",
    "strategy": "FSDP",
    "dtype": "BF16",
    "device_name": "H100",
    "bsz": 2
    },
    {
    "cfg_path": "configs/llama-3.1-8b.json",
    "strategy": "FSDP",
    "dtype": "FP8",
    "device_name": "H100",
    "bsz": 1
    },
    {
    "cfg_path": "configs/llama-3.1-70b-proxy4.json",
    "strategy": "FSDP",
    "dtype": "BF16",
    "device_name": "H100",
    "bsz": 2
    },
    {
    "cfg_path": "configs/llama-3.1-70b-proxy4.json",
    "strategy": "FSDP",
    "dtype": "FP8",
    "device_name": "H100",
    "bsz": 4
    },
    {
    "cfg_path": "configs/mistral-7b-v0.1.json",
    "strategy": "FSDP",
    "dtype": "BF16",
    "device_name": "H100",
    "bsz": 1
    },
    {
    "cfg_path": "configs/mistral-7b-v0.1.json",
    "strategy": "FSDP",
    "dtype": "FP8",
    "device_name": "H100",
    "bsz": 1
    }
]

H200_CONFIG = [
    {
    "cfg_path": "configs/gpt2-1.5b.json",
    "strategy": "DDP",
    "dtype": "BF16",
    "device_name": "H100",
    "bsz": 28
    },
    {
    "cfg_path": "configs/gpt2-1.5b.json",
    "strategy": "DDP",
    "dtype": "FP8",
    "device_name": "H100",
    "bsz": 38
    },
    {
    "cfg_path": "configs/llama-3.1-8b.json",
    "strategy": "FSDP",
    "dtype": "BF16",
    "device_name": "H100",
    "bsz": 4
    },
    {
    "cfg_path": "configs/llama-3.1-8b.json",
    "strategy": "FSDP",
    "dtype": "FP8",
    "device_name": "H100",
    "bsz": 4
    },
    {
    "cfg_path": "configs/llama-3.1-70b-proxy4.json",
    "strategy": "FSDP",
    "dtype": "BF16",
    "device_name": "H100",
    "bsz": 8
    },
    {
    "cfg_path": "configs/llama-3.1-70b-proxy4.json",
    "strategy": "FSDP",
    "dtype": "FP8",
    "device_name": "H100",
    "bsz": 8
    },
    {
    "cfg_path": "configs/mistral-7b-v0.1.json",
    "strategy": "FSDP",
    "dtype": "BF16",
    "device_name": "H100",
    "bsz": 2
    },
    {
    "cfg_path": "configs/mistral-7b-v0.1.json",
    "strategy": "FSDP",
    "dtype": "FP8",
    "device_name": "H100",
    "bsz": 2
    }
]


CONFIG = {
    "H100": H100_CONFIG
}

def main():
    if "H100" in torch.cuda.get_device_name():
        device_name = 'H100'
    elif "H200" in torch.cuda.get_device_name():
        device_name = 'H200'
    elif "MI300X" in torch.cuda.get_device_name():
        device_name = 'MI300X'
    else:
        raise ValueError(f'GPU device {torch.cuda.get_device_name()} not supported.')

    log_path = 'outputs/benchmark_results.csv'
    assert not Path(log_path).exists()
    with open(log_path, 'w') as f:
        f.write('Model, Strategy, GPU, dtype, Batch Size, TFLOP/s/GPU, MFU')

    for cfg_d in CONFIG[device_name]:
        benchmark(**cfg_d, log_path=log_path)
        time.sleep(1)


def benchmark(cfg_path, strategy, dtype, device_name, bsz, log_path, **kwargs):
    assert device_name in torch.cuda.get_device_name(), 'Incompatible GPU device for the benchmark config'

    with open(log_path, 'a') as f:
        f.write(f'\n{Path(cfg_path).stem}, {strategy}, {device_name}, {dtype}, {bsz}, ')

    train_fn_dict = {'Single': train_single, 'DDP': train_ddp, 'FSDP': train_fsdp}
    train_fn = train_fn_dict[strategy]

    try:
        train_fn(
            cfg_path=cfg_path,
            bsz=bsz,
            use_fp8=(dtype == 'FP8'),
            pt_compile=(dtype == 'BF16'),
            log_path=log_path,
            n_steps=64*8,
            **kwargs
        )
    except Exception as e:
        with open(log_path, 'a') as f:
            f.write('OOM, OOM')


if __name__ == '__main__':
    main()
