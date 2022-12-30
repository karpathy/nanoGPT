
# nanoGPT

The simplest, fastest repository for training/finetuning medium-sized GPTs. It's a re-write of [minGPT](https://github.com/karpathy/minGPT), which I think became too complicated, and which I am hesitant to now touch. Still under active development, currently working to reproduce GPT-2 on OpenWebText dataset. The code itself aims by design to be plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- `pip install datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `pip install tiktoken` for OpenAI's fast bpe code <3
- `pip install wandb` for optional logging <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

To my knowledge, running this with the current script with the GPT-2 hyperparameters should reproduce the GPT-2 result, provided that OpenWebText ~= WebText. I'd like to make the code more efficient before attempting to go there. Once some checkpoints are written to the output directory (e.g. `./out` by default), we can sample from the model:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

Training on 1 A100 40GB GPU overnight currently gets loss ~3.74, training on 4 gets ~3.60. Random chance at init is -ln(1/50257) = 10.82. Which brings us to baselines:

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
python train.py config/eval_gpt2.py
python train.py config/eval_gpt2_medium.py
python train.py config/eval_gpt2_large.py
python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

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
