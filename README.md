
# nanoGPT

The cleanest, fastest repository for training/finetuning medium-sized GPTs. Still under active development, currently trying to reproduce GPT-2 on OpenWebText dataset. The code itself is tiny, plain and readable. At the moment `train.py` is a ~200-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can also load the GPT-2 weights from OpenAI.

## install

We need a few dependencies:

- [pytorch](https://pytorch.org), of course
- numpy
- `pip install datasets` for huggingface datasets
- `pip install tiktoken` for OpenAI's fast bpe code
- `pip install wandb` for optional logging

## usage

To render a dataset we first tokenize some documents into one giant array of indices. E.g. for OpenWebText see:

```
$ cd data/openwebtext
$ python prepare.py
```

To download and tokenize the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. The training script currently by default tries to reproduce the smallest GPT-2 released by OpenAI, i.e. the 124M version of GPT-2. We can train as follows, though I encourage you to read the code and see all of the settings and paths up top in the file:

```
$ python train.py
```

Once some checkpoints are written to the output directory `out`, we can sample from the model:

```
$ python sample.py
```

Training on 1 GPU overnight currently gets loss ~3.74. Random chance at init is -ln(1/50257) = 10.82. Which brings us to baselines.

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
