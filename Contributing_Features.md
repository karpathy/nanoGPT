# How to add new features

This is a guide for adding a new feature to the search space.

# TOC

* [Step 1 Add a config within the model.py](#step-1-add-a-config-within-the-modelpy)
* [Step 2 Add an argparse argument for the train.py](#step-2-add-an-argparse-argument-for-the-trainpy)
* [Step 3 Create script in exploration folder](#step-3-create-script-in-exploration-folder)
* [Other Parameter Groups](#other-parameter-groups)
* [Ideas](#ideas)

## Step 1 Add a config within the model.py

Open up `model.py` and add your new configuration within the `GPTConfig`
dataclass:

```python
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

## Step 2 Add an argparse argument for the train.py


Open up `train.py` and add your new feature to the model group inside `parse_args` function,
depending on the type:

For boolean values:
```python
model_group.add_argument('--use_faster_inference', default=True, action=argparse.BooleanOptionalAction)
```

For string values (e.g. for selection between several types of a module):
```python
model_group.add_argument("--softmax_variant", type=str, default="softermax", choices=["constantmax", "polymax", "strongermax", "softermax", "sigsoftmax", "sigsoftmax_base2"])
```

For numeric values:
```python
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

dataset="shakespeare_char"
python3 "data/${dataset}/prepare.py"

softmax_variation=("constantmax" "polymax" "softermax" "sigsoftmax")

max_iters="3000"
block_size="256"
notes="check_all_softmax_variations"

# Loop over the array
for softmax_variant in "${softmax_variation[@]}"
do
  python3 train.py \
    --max_iters "$max_iters" \
    --eval_iters 200 \
    --eval_interval 200 \
    --log_interval 10 \
    --device cuda \
    --dataset "$dataset" \
    --use_softmax_variant \
    --softmax_variant "${softmax_variant}" \
    --use_softermax_xmax \
    --tensorboard_project "${dataset}_${softmax_variant}_${max_iters}" \
    --tensorboard_run_name "${softmax_variant}_${notes}" \
    --block_size "$block_size" \
    --out_dir "${dataset}_${softmax_variant}_${max_iters}_${notes}" \
    --compile
done
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

