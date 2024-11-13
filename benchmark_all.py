import json
import time
from pathlib import Path

import torch

from train import train as train_single
from train_ddp import train as train_ddp
from train_fsdp import train as train_fsdp


def main():
    device_name = 'H100'  # Options: 'H100', 'H200', 'MI300X'
    cfg_suffix = f'*_{device_name.lower()}.json'

    log_path = 'outputs/benchmark_results.csv'
    assert not Path(log_path).exists()
    with open(log_path, 'w') as f:
        f.write('Model, Strategy, GPU, dtype, Batch Size, TFLOP/s/GPU, MFU')

    for bm_cfg_path in sorted(Path('bm_configs/').glob(cfg_suffix)):
        with open(bm_cfg_path) as f:
            cfg_d = json.load(f)
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
