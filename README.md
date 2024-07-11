# ThunderKittens nanoGPT

This repository contains code for training GPT models with ThunderKittens CUDA kernels for NVidia H100 GPUs. We adapt the popular nanoGPT repository. 

## Setup

Create an environment and install nanoGPT dependencies:
```bash
conda create -n env python=3.11

pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Install ThunderKittens kernels:
```
git submodule init
git submodule update
cd ThunderKittens/
source env.src

cd ThunderKittens/examples/attn_causal/
python h100_fwd_setup.py build
```

Prepare the data, use the following command. We'll start with a character-level GPT on the works of Shakespeare. This creates a train.bin and val.bin in that data directory. 
```bash
python data/shakespeare_char/prepare.py
```

## Benchmark

Let's first benchmark the kernel to make sure that everything is set up correctly. Prepare `data` of choice from NanoGPT README below (modify ``dataset`` in `bench.py` path accordingly - default = `shakespeare_char`). 
To benchmark the TK Forward Causal Attention, set `TK_kernel` = True in `bench.py` and run:

```bash
python bench.py
```

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## Training and inference

We can train a full model using our kernels:
```bash
python train.py config/train_shakespeare_char.py
```

And run inference:
```bash
python sample.py --out_dir=out-shakespeare-char
```

To scale things up with an 8 GPU node:
```bash
python data/openwebtext/prepare.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

## Finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:
```bash
python train.py config/finetune_shakespeare.py
```

This will load the config parameter overrides in `config/finetune_shakespeare.py`. Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-shakespeare` by default, per the config file. 

## Inference with a pre-trained LM

Here is a script you can use to sample from the largest available `gpt2-medium` model with and without TK kernels. 
```bash
python inference.py
```

## Citations

We build on the [nanoGPT](https://github.com/karpathy/nanoGPT) repository. To learn more about the repository, please view the original README.
