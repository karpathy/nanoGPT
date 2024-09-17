# Experiment Reproduction

Install the minimal dataset and plotting requirements with `pip install -r requirements.txt`. We used the PyTorch NGC container for GPU-based runs, but any environment containing the dependencies from [the main README](https://github.com/EleutherAI/nanoGPT-mup?tab=readme-ov-file#install) will suffice.

To download the tiny shakespeare dataset, run `python data/shakespeare_char/prepare.py`. For OpenWebText (OWT), run `python data/openwebtext/prepare.py`.


# Coordinate Checks

The lowest-overhead correctness check of a mutransfer implementation is a [coordinate check](https://github.com/microsoft/mup?tab=readme-ov-file#checking-correctness-of-parametrization).

To run coordinate checks in our implementation using the tiny shakespeare dataset, use the following scripts for Standard Parameterization (SP):

```
bash mup_examples/coord_check_shakespeare_char/sp/run.sh
```

And muP:

```
bash mup_examples/coord_check_shakespeare_char/mup/run.sh
```

These scripts populate the `out/` subdirectories with your coord check data, which you can then plot with `mup_examples/coord_check_shakespeare_char/plot.ipynb`


# Learning Rate muTransfer

To actually test transferring hyperparameters, you need to run training for a set number of steps on a chosen dataset. 

1. Tiny Shakespeare is small and simple enough to see stable training loss with few iterations and small batch sizes, so we recommend it to test transfer quickly or on compute-constrained systems (e.g. laptop/desktop CPU). 
2. OpenWebText is comparatively large, but more representative of the massive webcrawl-based datasets used to train most models today. 

The default values chosen in each `run.sh` reflect this.

## Tiny Shakespeare

To sweep over seeds, model widths, and learning rates on the tiny shakespeare dataset with muP:

```
bash mup_examples/mutransfer_lr_shakespeare_char/mup/run.sh
```

and SP:

```
bash mup_examples/mutransfer_lr_shakespeare_char/sp/run.sh
```

## OpenWebText

To sweep over seeds, model widths, and learning rates on the OpenWebText (OWT) dataset with muP:

```
bash mup_examples/mutransfer_lr_owt/mup/run.sh
```

and SP:

```
bash mup_examples/mutransfer_lr_owt/sp/run.sh
```

These scripts populate 

These scripts populate the `out/` subdirectories with your train loss data, which you can then plot with `mup_examples/mutransfer_lr/plot.ipynb`