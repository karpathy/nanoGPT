# How to add new features

This is a guide for adding a new feature to the search space.

# TOC

* [Step 1  Add a config within the model.py](#step-1-add-a-config-within-the-model.py)
* [Step 2  Add an argparse argument for the train.py](#step-2-add-an-argparse-argument-for-the-train.py)
* [Step 3 Create script in exploration folder](#step-3-create-script-in-exploration-folder)
* [Other Parameter Groups](#other-parameter-groups)
* [Ideas](#ideas)

## Step 1  Add a config within the model.py

Open up `model.py` and add your new configuration within the `GPTConfig`
dataclass:

```
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest m
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

    # Your New Feature
    use_faster_inference: bool = True
```

## Step 2  Add an argparse argument for the train.py


Open up `train.py` and add your new feature to the model group inside `parse_args` function,
depending on the type:

For boolean values:
```
model_group.add_argument('--use_faster_inference', default=True, action=argparse.BooleanOptionalAction)
```

For string values (e.g. for selection between several types of a module):
```
model_group.add_argument("--softmax_variant", type=str, default="softermax", choices=["constantmax", "polymax", "strongermax", "softermax", "sigsoftmax", "sigsoftmax_base2"])
```

For numeric values:
```
model_group.add_argument("--block_size", type=int, default=256)
```



## Step 3 Create script in exploration folder


`cd` into the exploration folder and script in an exploration sweep.

This will automatically timestamp and apply labels to your tensorboard logs.

The following example shows testing with different types of softmax variations
on the default dataset:

```bash
#/bin/bash

# head to repo root
cd ../

# create train.bin and val.bin splits (retaining contiguous sections of data)
python3 data/shakespeare_char/prepare.py

# start training
python3 train.py \
  --max_iters 3000 \
  --eval_iters 200 \
  --eval_interval 200 \
  --log_interval 10 \
  --use_softmax_variant \
  --dataste
  --softmax_variant "constantmax" \
  --tensorboard_project "softmax_explorations" \
  --tensorboard_run_name "consmax_base_e" \
  --block_size 256  \
  --out_dir "consmax_evaluations" \
  --compile

# start training
python3 train.py \
  --max_iters 3000 \
  --eval_iters 200 \
  --eval_interval 200 \
  --log_interval 10 \
  --use_softmax_variant \
  --softmax_variant "softermax" \
  --tensorboard_project "softmax_explorations" \
  --tensorboard_run_name "softermax" \
  --block_size 256  \
  --out_dir "softermax_evaluations" \
  --compile

# start training
python3 train.py \
  --max_iters 3000 \
  --eval_iters 200 \
  --eval_interval 200 \
  --log_interval 10 \
  --no-use_softmax_variant \
  --tensorboard_project "softmax_explorations" \
  --tensorboard_run_name "regular_softmax" \
  --block_size 256  \
  --out_dir "softmax_evaluations" \
  --compile
```

## Other Parameter Groups

`train.py` is parameterized with argparse into three groups:

1. `model_group` - these are automatically added to a config and sent to model.py
2. `training_group` - only used by train.py
3. `logging_group` - also only used by train.py (specifically for logging)

Adding to the model group will have it sent into model.py, making it really just
a two step process for adding a new feature.

## Ideas

In addition to scanning perplexity results from for different settings:

- Reinforcement Loops - adding gymnasium to optimize parameters
- Training Loop - Generating output sample.py, augmenting, then feeding back as training data.
- Monitoring of hyperparameters - e.g. gamma and beta values for constantmax

