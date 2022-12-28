
# nanoGPT

The cleanest, fastest repository for training/finetuning medium-sized GPTs.

This repo currently requires reading the code, but it's not that bad. work ongoing...

Getting started:

We need a few dependencies:

- [pytorch](https://pytorch.org), of course
- numpy
- `pip install datasets` for huggingface datasets
- `pip install tiktoken` for OpenAI's fast bpe code
- `pip install wandb` for optional logging

Then we want to render the detaset:

```
$ cd data/openwebtext
$ python prepare.py
```

To download and tokenize the [openwebtext](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in a massive sequence. Then we're ready to kick off training. The training script currently tries to reproduce the smallest GPT-2 released by OpenAI, i.e. the 124M version of GPT-2. We can run it like so:

```
$ python train.py
```

Once some checkpoints are written to the output directory `out`, we're ready to sample from the model:

```
$ python sample.py
```

