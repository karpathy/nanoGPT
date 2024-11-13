import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import RootModel

from train import train as train_single
from train_ddp import train as train_ddp
from train_fsdp import train as train_fsdp


@dataclass
class BenchmarkConfig:
    cfg_path: str
    strategy: str
    dtype: str
    device_name: str
    bsz: int

    def __post_init__(self):
        assert Path(self.cfg_path).exists()
        assert self.strategy in ['Single', 'DDP', 'FSDP']
        assert self.dtype in ['BF16', 'FP8']
        assert any(dev in self.device_name for dev in ['H100', 'H200', 'MI300X'])


def main():
    model_names = ['gpt2-1.5b', 'llama-3.1-70b-proxy', 'llama-3.1-8b', 'llama-2-7b', 'mistral-7b-v0.1']
    dtypes = ['BF16', 'FP8']
    device_name = 'H100'  # Options: 'H100', 'H200', 'MI300X'

    log_path = 'bm_configs/config_search.csv'
    assert not Path(log_path).exists()
    with open(log_path, 'w') as f:
        f.write('Model, Strategy, GPU, dtype, Batch Size, TFLOP/s/GPU, MFU')

    for model_name in model_names:
        if model_name.startswith('gpt2'):
            strategy = 'DDP'
            batch_sizes = [10, 12, 14, 16]
        else:
            strategy = 'FSDP'
            batch_sizes = [1, 2, 4, 8]

        for dtype in dtypes:
            search_config(model_name, strategy, dtype, device_name, batch_sizes, log_path)
            time.sleep(10)


def search_config(model_name, strategy, dtype, device_name, batch_sizes, log_path, **kwargs):
    train_fn_dict = {'Single': train_single, 'DDP': train_ddp, 'FSDP': train_fsdp}
    train_fn = train_fn_dict[strategy]
    max_bsz = 1

    for bsz in batch_sizes:
        with open(log_path, 'a') as f:
            f.write(f'\n{model_name}, {strategy}, {torch.cuda.get_device_name()}, {dtype}, {bsz}, ')

        try:
            train_fn(
                cfg_path=f'configs/{model_name}.json',
                bsz=bsz,
                use_fp8=(dtype == 'FP8'),
                pt_compile=(dtype == 'BF16'),
                log_path=log_path,
                n_steps=64*8,
                **kwargs
            )
            max_bsz = bsz
            time.sleep(1)
        except:
            with open(log_path, 'a') as f:
                f.write('OOM, OOM')
            break

    root_model = RootModel[BenchmarkConfig](BenchmarkConfig(
        cfg_path=f'configs/{model_name}.json',
        strategy=strategy,
        dtype=dtype,
        device_name='H100',
        bsz=max_bsz
    ))
    with open(f'bm_configs/{model_name}_{strategy}_{dtype}_{device_name}.json'.lower(), 'w') as f:
        json.dump(root_model.model_dump(), f, indent=2)


if __name__ == '__main__':
    main()
