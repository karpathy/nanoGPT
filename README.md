# Benchmark LLM Training

Repo for benchmarking LLM training workloads.


## Installation

To build the image:
```bash
docker build -t llm-train-bench .
```

To start the container:
```bash
./docker/launch.sh
```


## Single GPU Training

```
NAME
    train.py

SYNOPSIS
    train.py CFG_PATH <flags>

POSITIONAL ARGUMENTS
    CFG_PATH
        Type: str

FLAGS
    --gpu_id=GPU_ID
        Type: int
        Default: 0
    -b, --bsz=BSZ
        Type: int
        Default: 8
    --n_workers=N_WORKERS
        Type: int
        Default: 8
    --n_steps=N_STEPS
        Type: int
        Default: 128
    --grad_acc_steps=GRAD_ACC_STEPS
        Type: int
        Default: 8
    -c, --ckpt_freq=CKPT_FREQ
        Type: int
        Default: 64
    --pt_compile=PT_COMPILE
        Type: bool
        Default: False
    --profile=PROFILE
        Type: bool
        Default: False
    -o, --output_dir=OUTPUT_DIR
        Type: str
        Default: 'outputs/single_gpu'

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```


## Multi-GPU Training with DDP

```
NAME
    train_ddp.py

SYNOPSIS
    train_ddp.py CFG_PATH <flags>

POSITIONAL ARGUMENTS
    CFG_PATH
        Type: str

FLAGS
    -b, --bsz=BSZ
        Type: int
        Default: 8
    --n_workers=N_WORKERS
        Type: int
        Default: 8
    --n_steps=N_STEPS
        Type: int
        Default: 1024
    -g, --grad_acc_steps=GRAD_ACC_STEPS
        Type: int
        Default: 8
    -c, --ckpt_freq=CKPT_FREQ
        Type: int
        Default: 64
    --pt_compile=PT_COMPILE
        Type: bool
        Default: False
    --profile=PROFILE
        Type: bool
        Default: False
    -o, --output_dir=OUTPUT_DIR
        Type: str
        Default: 'outputs/ddp/'

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENTS
```
