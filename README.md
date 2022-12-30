
# nanoGPT

The simplest, fastest repository for training/finetuning medium-sized GPTs. It's a re-write of [minGPT](https://github.com/karpathy/minGPT), which I think became too complicated, and which I am hesitant to now touch. Still under active development, currently working to reproduce GPT-2 on OpenWebText dataset. The code itself aims by design to be plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

## install

Dependencies:

- [pytorch](https://pytorch.org) <3
- `pip install datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `pip install tiktoken` for OpenAI's fast bpe code <3
- `pip install wandb` for optional logging <3
- `pip install tqdm`

## usage

To render a dataset we first tokenize some documents into one simple long 1D array of indices. E.g. for OpenWebText see:

```
$ cd data/openwebtext
$ python prepare.py
```

To download and tokenize the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. This will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. The training script currently by default tries to reproduce the smallest GPT-2 released by OpenAI, i.e. the 124M version of GPT-2. We can demo train as follows on a single device, though I encourage you to read the code and see all of the settings and paths up top in the file:

```
$ python train.py
```

To train using PyTorch Distributed Data Parallel (DDP) run the script with torchrun. For example to train on a node with 4 GPUs run:

```
$ torchrun --standalone --nproc_per_node=4 train.py
```

To my knowledge, running this with the current script with the GPT-2 hyperparameters should reproduce the GPT-2 result, provided that OpenWebText ~= WebText. I'd like to make the code more efficient before attempting to go there. Once some checkpoints are written to the output directory (e.g. `./out` by default), we can sample from the model:

```
$ python sample.py
```

Training on 1 A100 40GB GPU overnight currently gets loss ~3.74, training on 4 gets ~3.60. Random chance at init is -ln(1/50257) = 10.82. Which brings us to baselines:

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```
$ python train.py eval_gpt2
$ python train.py eval_gpt2_medium
$ python train.py eval_gpt2_large
$ python train.py eval_gpt2_xl
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

I briefly tried finetuning gpt2 a bit more on our OWT and didn't notice dramatic improvements, suggesting that OWT is not much much different from WT in terms of the data distribution, but this needs a bit more thorough attempt once the code is in a better place.

## benchmarking

For model benchmarking `bench.py` might be useful. It's identical what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

## efficiency notes

Code by default now uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

A few that I'm aware of, other than the ones mentioned in code:

- Additional optimizations to the running time
- Report and track other metrics e.g. PPL
- Eval zero-shot perplexities on PTB, WikiText, other related benchmarks
- Current initialization (PyTorch default) departs from GPT-2. In a very quick experiment I found it to be superior to the one suggested in the papers, but that can't be right
- Currently fp16 is much faster than bf16. Potentially revert back to using fp16 and re-introduce the gradient scaler?
- Add some finetuning dataset and guide on some dataset for demonstration.
- Reproduce GPT-2 results. It was estimated ~3 years ago that the training cost of 1.5B model was ~$50K
