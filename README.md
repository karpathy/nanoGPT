### Abstract

Multi Head Latent Attention Uses SVD for matrix compression. Our goal it to implement randomized-SVD instead of SVD, and make the process more efficient.

### Standard SVD Implementation

[Standard SVD README](https://github.com/cangokmen/CS599-Randomized-SVD/blob/master/docs/README(SVD).md)

### How to run?
There are multiple versions of the GPT implementation in this repository. 

In any case you are required to install the dependencies:

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

and for testing purposes you can prepare a small dataset as follows:



**1. In order to run regular attention version:**

Train it 
```sh
python -m mla_gpt.cli.train config/train_shakespeare_char.py \
    --device=cpu \
    --compile=False \
    --eval_iters=20 \
    --log_interval=1 \
    --block_size=64 \
    --batch_size=12 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=2000 \
    --lr_decay_iters=2000 \
    --dropout=0.0 \
    --dtype=float32
```
and run to see a sample output:

```sh
python sample.py --out_dir=out-shakespeare-char
```


**2. In order to run Multi-Head Latent Attention with Regular SVD version:**


**3. In order to run Multi-Head Latent Attention with Randomized SVD version:**






### acknowledgements
MLA implementation is based on nanoGPT implementation of Karpathy.
https://github.com/karpathy/nanoGPT
