# How to add new features

This is a guide for adding a new feature to the search space.

# TOC

* [Step 1 Add new variation](#step-1-add-new-variation)
* [Step 2 Adjust model.py](#step-2-adjust-modelpy)
* [Step 3 Add a config within the model.py](#step-3-add-a-config-within-the-modelpy)
* [Step 4 Add an argparse argument for the train.py](#step-4-add-an-argparse-argument-for-the-trainpy)
* [Step 5 Create configuration json in exploration folder](#step-5-create-configuration-json-in-exploration-folder)
* [Other Parameter Groups](#other-parameter-groups)
* [Ideas](#ideas)

## Step 1 Add new variation


If the variation is in the following categories, add to the appropriate file or
create a new file in the variations folder:

```
variations/
├── activation_variations.py
├── normalization_variations.py
├── position_encoding_variations.py
└── softmax_variations.py
```
Some variations, such as orderings of the network, may need to be made directly
to the `model.py` file.

## Step 2 Adjust model.py

Import the new variation:
```
from variations.softmax_variations import YourSoftmaxVariation
```

And add to the model.py in appropriate section:
```
    if self.softmax_variant_attn == "yournewvariation":
      self.softmax_layer = YourNewVariation(config)
```

## Step 3 Add a config within the model.py

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

    # Your New Setting
    new_variation_setting: bool = True
```

## Step 4 Add an argparse argument for the train.py


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

## Step 5 Create configuration json in exploration folder

`cd` into the exploration folder and copy a template for a new exploration sweep.

Run the sweep with `run_experiments.py` from the repo root specifying our
config file.

```
python3 run_experiments.py --config explorations/config.json --output_dir out_test
```

This will automatically timestamp and apply labels to your tensorboard logs,
create direct csv logs, and save output checkpoints into a specified folder.

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

