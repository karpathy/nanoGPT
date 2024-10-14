# Benchmark LLM Training

Repo for benchmarking LLM training workloads.


## TODO
- [ ] te.checkpoint @kimbo
- [ ] debug AMD TE @orenleung
- [ ] add perf to spreadsheet @orenleung
- [ ] H200 & mi300x perf to spreadsheet @orenleung
- [ ] multi-node nvidia @orenleung
- [ ] multi-node amd @orenleung


## Setup

### NVIDIA

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
docker build -t llm-train-bench/amd -f ./docker/Dockerfile.amd .
```

To start the container:
```bash
./docker/launch_amd.sh
```

> [!IMPORTANT]
> set env flag `DISABLE_ADDMM_HIP_LT=0`


## Usage

### Example Usage

Single GPU:
```bash
python train.py configs/gpt2-125m.json --pt-compile --compile-mode "reduce-overhead"
```

DDP:
```bash
python train_ddp.py configs/gpt2-1.5b.json --bsz 32 --use-fp8
```

FSDP:
```bash
python train_fsdp.py configs/llama-3.1-70b-proxy.json --sac-freq 1/4
```

### Feature Support

| Feature | Single GPU | DDP | FSDP |
| :- | :-: | :-: | :-: |
| Mixed-precision Training | V | V | V |
| Transformer Engine FP8 | V | V | V |
| PyTorch Compile | V | V | V |
| Selective Activation Checkpointing | X | X | V |


### Feature Compatibility

Single GPU and DDP:

| | Mixed-precision Training | PyTorch Compile | FP8 |
| -: | :-: | :-: | :-: |
| Mixed-precision Training | | V | V |
| PyTorch Compile | V | | X |
| FP8 | V | X | |

FSDP:

| | Mixed-precision Training | PyTorch Compile | FP8 | Selective AC |
| -: | :-: | :-: | :-: | :-: |
| Mixed-precision Training | | V | V | V |
| PyTorch Compile | V | | X | V |
| FP8 | V | X | | X |
| Selective AC | V | V | X | |


## Performance Results

- MP: Mixed-precision Training (bfloat16)
- PC: PyTorch Compile (default mode)
- SAC: Selective Activation Checkpointing (1 every 4 blocks)


### NVIDIA H100

| | `gpt2-1.5b` | `llama-3.1-70b-proxy` |
| :-: | :-: | :-: |
| Batch Size | 8 | 2 |

Single GPU:
| Config | `gpt2-1.5b` | `llama-3.1-70b-proxy` |
| :- | :-: | :-: |
| MP | 350 TFLOP/s (35% MFU) | 700 TFLOP/s (70% MFU) |
| MP + PC | 400 TFLOP/s (40% MFU) | 790 TFLOP/s (80% MFU) |
| MP + FP8 | 400 TFLOP/s (20% MFU) | 1200 TFLOP/s (60% MFU) |

DDP:
| Config | `gpt2-1.5b` | `llama-3.1-70b-proxy` |
| :- | :-: | :-: |
| MP | 325 TFLOP/s (32% MFU) | 600 TFLOP/s (60% MFU) |
| MP + PC | 360 TFLOP/s (36% MFU) | 650 TFLOP/s (65% MFU) |
| MP + FP8 | 350 TFLOP/s (20% MFU) | 950 TFLOP/s (48% MFU) |

FSDP:
| Config | `gpt2-1.5b` | `llama-3.1-70b-proxy` |
| :- | :-: | :-: |
| MP | 320 TFLOP/s (32% MFU) | 650 TFLOP/s (65% MFU) |
| MP + PC | 330 TFLOP/s (33% MFU) | 680 TFLOP/s (68% MFU) |
| MP + FP8 | 250 TFLOP/s (12% MFU) | 1000 TFLOP/s (50% MFU) |

> [!NOTE]
> FP8 seems to allow larger batch sizes. Will update the numbers in the future.


#### Selective AC

Current selective AC implementation works only when there is more than 1 transformer block,
so we replace `llama-3.1-70b-proxy` with `llama-3.1-8b`.

| | `gpt2-1.5b` | `llama-3.1-8b` |
| :-: | :-: | :-: |
| Selective AC | 1/8 | 1/4 |

| Config | `gpt2-1.5b` | `llama-3.1-8b` |
| :- | :-: | :-: |
| MP + SAC | 320 TFLOP/s (32% MFU) | 330 TFLOP/s (33% MFU) |
| MP + SAC + PC | 330 TFLOP/s (33% MFU) | Crashes |


### AMD MI300X

`/long_pathname_so_that_rpms_can_package_the_debug_info/src/external/clr/hipamd/src/hip_internal.hpp:555`


## References

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [pytorchlabs/gpt-fast](https://github.com/pytorch-labs/gpt-fast/tree/main)
- [Multi-GPU training with DDP](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)
- [PyTorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [`torch.profiler` docs](https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile)
- [Tutorial: Getting started with FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Tutorial: Advanced model training with FSDP](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html)
