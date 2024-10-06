# Benchmark LLM Training

Repo for benchmarking LLM training workloads.

## TODO: 
- [x] reprod why torch compile doesnt work on amd pypi nightly even on gpt2 single gpu @orenleung
- [ ] selection act checkpointing @kimbochen
- [ ] removing torch.save 
- [ ] benchmark fsdp + selection act ckpt 8B llama3 @kimbochen
- [ ] multi-node nvidia
- [ ] multi-node amd


## Installation

To build the image:
```bash
docker build -t llm-train-bench -f ./docker/Dockerfile .
```

To start the container:
```bash
./docker/launch.sh
```
### AMD
To build the image:
```bash
docker build -t llm-train-bench -f ./docker/Dockerfile.amd .
```

To start the container:
```bash
./docker/launch_amd.sh
```

##### IMPORTANT for amd
set env flag `DISABLE_ADDMM_HIP_LT=0`


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


## References

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [pytorchlabs/gpt-fast](https://github.com/pytorch-labs/gpt-fast/tree/main)
- [Multi-GPU training with DDP](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)
- [PyTorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [`torch.profiler` docs](https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile)
- [Tutorial: Getting started with FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Tutorial: Advanced model training with FSDP](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html)
