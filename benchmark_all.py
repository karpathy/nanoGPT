import torch

import train
import train_ddp
import train_fsdp


def main(output_fname):
    with open(f'{output_fname}.csv', 'w') as f:
        f.write('Model, Strategy, GPU, dtype, TFLOP/s/GPU, MFU')

    bench_train = catch_error_and_continue(train.train, output_fname)
    bench_train_ddp = catch_error_and_continue(train_ddp.train, output_fname)
    bench_train_fsdp = catch_error_and_continue(train_fsdp.train, output_fname)


    # GPT2 1.5B DDP bf16
    cfg = dict(
        cfg_path='configs/gpt2-1.5b.json',
        bsz=8,
        pt_compile=True,
        bench_fname=output_fname
    )
    write_config_csv(output_fname, cfg, 'DDP')
    bench_train_ddp(**cfg)


    # GPT2 1.5B DDP fp8
    cfg = dict(
        cfg_path='configs/gpt2-1.5b.json',
        bsz=8,
        use_fp8=True,
        bench_fname=output_fname
    )
    write_config_csv(output_fname, cfg, 'DDP')
    bench_train_ddp(**cfg)


    # LLaMA 3.1 70B Proxy 4 FSDP bf16
    cfg = dict(
        cfg_path='configs/llama-3.1-70b-proxy4.json',
        bsz=4,
        pt_compile=True,
        bench_fname=output_fname
    )
    write_config_csv(output_fname, cfg, 'FSDP')
    bench_train_fsdp(**cfg)


    # LLaMA 3.1 70B Proxy 4 FSDP fp8
    cfg = dict(
        cfg_path='configs/llama-3.1-70b-proxy4.json',
        bsz=4,
        use_fp8=True,
        bench_fname=output_fname
    )
    write_config_csv(output_fname, cfg, 'FSDP')
    bench_train_fsdp(**cfg)


    # LLaMA 3.1 8B FSDP bf16
    cfg = dict(
        cfg_path='configs/llama-3.1-8b.json',
        bsz=2,
        pt_compile=True,
        bench_fname=output_fname
    )
    write_config_csv(output_fname, cfg, 'FSDP')
    bench_train_fsdp(**cfg)


    # LLaMA 3.1 8B FSDP fp8
    cfg = dict(
        cfg_path='configs/llama-3.1-8b.json',
        bsz=1,
        use_fp8=True,
        bench_fname=output_fname
    )
    write_config_csv(output_fname, cfg, 'FSDP')
    bench_train_fsdp(**cfg)


    # LLaMA 2 7B FSDP bf16
    cfg = dict(
        cfg_path='configs/llama-2-7b.json',
        bsz=2,
        pt_compile=True,
        bench_fname=output_fname
    )
    write_config_csv(output_fname, cfg, 'FSDP')
    bench_train_fsdp(**cfg)


    # LLaMA 2 7B FSDP fp8
    cfg = dict(
        cfg_path='configs/llama-2-7b.json',
        bsz=2,
        use_fp8=True,
        bench_fname=output_fname
    )
    write_config_csv(output_fname, cfg, 'FSDP')
    bench_train_fsdp(**cfg)


    # Mistral v0.1 7B FSDP bf16
    cfg = dict(
        cfg_path='configs/mistral-v0.1.json',
        bsz=1,
        pt_compile=True,
        bench_fname=output_fname
    )
    write_config_csv(output_fname, cfg, 'FSDP')
    bench_train_fsdp(**cfg)


    # Mistral v0.1 7B FSDP fp8
    cfg = dict(
        cfg_path='configs/mistral-v0.1.json',
        bsz=1,
        use_fp8=True,
        bench_fname=output_fname
    )
    write_config_csv(output_fname, cfg, 'FSDP')
    bench_train_fsdp(**cfg)


def write_config_csv(filename, cfg, strategy):
    device_name = torch.cuda.get_device_name()
    model_name = cfg['cfg_path'].split('/')[-1]
    dtype = 'fp8' if cfg.get('use_fp8', False) else 'bf16' 

    with open(f'{filename}.csv', 'a') as f:
        f.write(f'\n{model_name}, {strategy}, {device_name}, {dtype}, ')


def catch_error_and_continue(fn, filename):
    def inner(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except:
            with open(f'{filename}.csv', 'a') as f:
                f.write('OOM, OOM')
            return
    return inner


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
