# GPT-2 XL
Sam Foreman
[<span class="orcid-green">{{< ai orcid >}}</span>](https://orcid.org/0000-0002-9981-0876)
2023-11-15

<span style="text-align:left;">[![Open In
Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saforem2/nanoGPT/blob/master/notebooks/ngpt-gpt2-medium.ipynb)</span>

## Install / Setup

### First Time Running

We need to install `ngpt` and setup the Shakespeare dataset

This will need to be ran the first time you are running this notebook.

Following the

``` python
!python3 -m pip install nanoGPT
```

you will need to restart your runtime (Runtime -\> Restart runtime)

After this, you should be able to

``` python
>>> import ngpt
>>> ngpt.__file__
'/content/nanoGPT/src/ngpt/__init__.py'
```

``` bash
%%bash

python3 -c 'import ngpt; print(ngpt.__file__)' 2> '/dev/null'

if [[ $? -eq 0 ]]; then
    echo "Has ngpt installed. Nothing to do."
else
    echo "Does not have ngpt installed. Installing..."
    git clone 'https://github.com/saforem2/nanoGPT'
    python3 nanoGPT/data/shakespeare_char/prepare.py
    python3 -m pip install -e nanoGPT -vvv
fi
```

<div class="cell-output cell-output-display">

    /lus/grand/projects/datascience/foremans/locations/thetaGPU/projects/saforem2/nanoGPT/src/ngpt/__init__.py
    Has ngpt installed. Nothing to do.

</div>

## Post Install

If installed correctly, you should be able to:

``` python
>>> import ngpt
>>> ngpt.__file__
'/path/to/nanoGPT/src/ngpt/__init__.py'
```

``` python
%load_ext autoreload
%autoreload 2

import ngpt
from rich import print
print(ngpt.__file__)
```

<div class="cell-output cell-output-display">

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:monospace">
&#10;    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
&#10;    <span style="color: #800080; text-decoration-color: #800080">/lus/grand/projects/datascience/foremans/locations/thetaGPU/projects/saforem2/nanoGPT/src/ngpt/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">__init__.py</span>
&#10;</pre>

</div>

## Build Trainer

Explicitly, we:

1.  `setup_torch(...)`
2.  Build `cfg: DictConfig = get_config(...)`
3.  Instnatiate `config: ExperimentConfig = instantiate(cfg)`
4.  Build `trainer = Trainer(config)`

``` python
import os
import numpy as np
from ezpz import setup_torch
from hydra.utils import instantiate
from ngpt.configs import get_config, PROJECT_ROOT
from ngpt.trainer import Trainer
from enrich.console import get_console

console = get_console()
HF_DATASETS_CACHE = PROJECT_ROOT.joinpath('.cache', 'huggingface')
HF_DATASETS_CACHE.mkdir(exist_ok=True, parents=True)

os.environ['MASTER_PORT'] = '5127'
os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE.as_posix()

SEED = np.random.randint(2**32)
console.print(f'SEED: {SEED}')

rank = setup_torch('DDP', seed=1234)
cfg = get_config(
    [
        'data=owt',
        'model=gpt2_xl',
        'optimizer=gpt2_xl',
        'train=gpt2_xl',
        'train.init_from=gpt2-xl',
        'train.max_iters=100',
        'train.dtype=bfloat16',
    ]
)
config = instantiate(cfg)
trainer = Trainer(config)
```

<div class="cell-output cell-output-display">

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:monospace,Menlo,'DejaVu Sans Mono',consolas,'Courier New'">
&#10;    --------------------------------------------------------------------------
    WARNING: There was an error initializing an OpenFabrics device.
    &#10;      Local host:   thetagpu24
      Local device: mlx5_0
    --------------------------------------------------------------------------
&#10;    SEED: <span style="color: #800080; text-decoration-color: #800080">125313342</span>
    RANK: 0 / 0
    <span style="color:#838383">[2023-11-10 17:36:01]</span><span style="color:#FD971F">[WARNING]</span><span style="color:#777777">[configs.py:298</span><span style="color:#777777">]</span> - No meta.pkl found, assuming GPT-<span style="color:#A0A">2</span> encodings<span style="color:#FD971F">...</span>
    <span style="color:#838383">[2023-11-10 17:36:01]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[configs.py:264</span><span style="color:#777777">]</span> - Rescaling GAS -> GAS <span style="color:#0A0">/</span><span style="color:#0A0">/</span> WORLD_SIZE = <span style="color:#A0A">1</span> <span style="color:#0A0">/</span><span style="color:#0A0">/</span> <span style="color:#A0A">1</span>
    <span style="color:#838383">[2023-11-10 17:36:01]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[configs.py:399</span><span style="color:#777777">]</span> - Tokens per iteration: <span style="color:#A0A">1</span>,<span style="color:#A0A">024</span>
    <span style="color:#838383">[2023-11-10 17:36:01]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[configs.py:431</span><span style="color:#777777">]</span> - Using <b><</b><b><span style="color:#F5F">torch.amp.autocast_mode.autocast</span></b><span style="color:#FFF"> object at </span><span style="color:#A0A">0x7f98e0139660</span><b>></b>
    <span style="color:#838383">[2023-11-10 17:36:01]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[trainer.py:184</span><span style="color:#777777">]</span> - Initializing from OpenAI GPT-<span style="color:#A0A">2</span> Weights: gpt2-xl
    2023-11-10 17:36:01.777923: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    [2023-11-10 17:36:05,925] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    <span style="color:#838383">[2023-11-10 17:36:06]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[model.py:225</span><span style="color:#777777">]</span> - loading weights from pretrained gpt: gpt2-xl
    <span style="color:#838383">[2023-11-10 17:36:06]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[model.py:234</span><span style="color:#777777">]</span> - forcing <i><span style="color:#55F">vocab_size</span></i>=<span style="color:#A0A">50257</span>, <i><span style="color:#55F">block_size</span></i>=<span style="color:#A0A">1024</span>, <i><span style="color:#55F">bias</span></i>=<i><span style="color:#5F5">True</span></i>
    <span style="color:#838383">[2023-11-10 17:36:06]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[model.py:240</span><span style="color:#777777">]</span> - overriding dropout rate to <span style="color:#A0A">0.0</span>
    <span style="color:#838383">[2023-11-10 17:36:29]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[model.py:160</span><span style="color:#777777">]</span> - number of parameters: <span style="color:#A0A">1555.</span>97M
    <span style="color:#838383">[2023-11-10 17:36:56]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[model.py:290</span><span style="color:#777777">]</span> - num decayed parameter tensors: <span style="color:#A0A">194</span>, with <span style="color:#A0A">1</span>,<span style="color:#A0A">556</span>,<span style="color:#A0A">609</span>,<span style="color:#A0A">600</span> parameters
    <span style="color:#838383">[2023-11-10 17:36:56]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[model.py:291</span><span style="color:#777777">]</span> - num non-decayed parameter tensors: <span style="color:#A0A">386</span>, with <span style="color:#A0A">1</span>,<span style="color:#A0A">001</span>,<span style="color:#A0A">600</span> parameters
    <span style="color:#838383">[2023-11-10 17:36:56]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[model.py:297</span><span style="color:#777777">]</span> - using fused AdamW: <i><span style="color:#5F5">True</span></i>
&#10;</pre>

</div>

## Prompt (**prior** to training)

``` python
query = "What is a supercomputer?"
outputs = trainer.evaluate(query, num_samples=1, display=False)
console.print(fr'\[prompt]: "{query}"')
console.print("\[response]:\n\n" + fr"{outputs['0']['raw']}")
```

<div class="cell-output cell-output-display">

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:monospace">
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
&#10;
&#10;
    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer? When it comes to massive computing, a supercomputer is simply a large computer system that has the ability to perform many calculations at once. This can be the result of using many different processing cores, or memory, or operating at a high clock speed. Supercomputers are often used to crack complex calculations and research problems.
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    On a larger scale, these massive computers are used to solve tough mathematical equations and solve hard scientific problems. They are very powerful enough to emulate the workings of the human brain and simulate a human intelligence in a virtual world.
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    In <span style="color: #800080; text-decoration-color: #800080">1992</span>, IBM's NeXTStep supercomputer was the largest and most powerful supercomputer in the world. It was released in <span style="color: #800080; text-decoration-color: #800080">1995</span> and did not continue to live up to its original promises, because its capabilities were quickly surpassed by its competitors.
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia
&#10;    Image credit: Wikipedia<span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">|endoftext|</span><span style="font-weight: bold">&gt;</span>Editor's note: Dan De Luce is the author of <span style="color: #008000; text-decoration-color: #008000">"When the Going Gets Tough: The New Survival Guide for College Students and Your Health and Well-Being."</span>
&#10;    College has never been more expensive. But with so many choices and so many choices of where to go, it's harder than ever for prospective students to find a college that fits their lifestyle.
&#10;    This is a problem—not just because it can be a hassle to find a college that doesn't require a large amount of financial aid. It's a problem because it can be costly for students to stay in college.
&#10;    So I created this list of colleges with the highest tuition where
&#10;</pre>

</div>

<div id="tbl-legend">

|  Name  | Description                   |
|:------:|:------------------------------|
| `step` | *Current training step*       |
| `loss` | *Loss value*                  |
|  `dt`  | *Time per step* (in **ms**)   |
| `sps`  | *Samples per second*          |
| `mtps` | *(million) Tokens per sec*    |
| `mfu`  | *Model Flops utilization*[^1] |

Table 1: Legend

</div>

## Train Model

``` python
trainer.model.module.train()
trainer.train()
```

<div class="cell-output cell-output-display">

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:monospace,Menlo,'DejaVu Sans Mono',consolas,'Courier New'">
&#10;    <span style="color:#838383">[2023-11-10 17:41:58]</span><span style="color:#2196F3">[INFO]</span><span style="color:#777777">[trainer.py:540</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">100</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.505</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">922</span><span style="color:#A0A">.295</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">1</span><span style="color:#A0A">.084</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.001</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">43</span><span style="color:#A0A">.897</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.555</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.558</span>
&#10;</pre>

</div>

## Evaluate Model

``` python
query = "What is a supercomputer?"
outputs = trainer.evaluate(query, num_samples=1, display=False)
```

``` python
from rich.text import Text
from enrich.console import get_console
console = get_console()

console.print(fr'\[prompt]: "{query}"')
console.print("\[response]:\n\n" + fr"{outputs['0']['raw']}")
```

<div class="cell-output cell-output-display">

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:monospace">
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer? A supercomputer is a machine that is exponentially more powerful than previous computing models while being far more energy efficient.
&#10;    What is an artificial neural network? An artificial neural network <span style="font-weight: bold">(</span>ANN<span style="font-weight: bold">)</span> is an order of magnitude more powerful than previous computational models, but has the same energy efficiency.
&#10;    For this article I will be using a machine learning technique called Backward-Compatible Neural Networks <span style="font-weight: bold">(</span>BCNNs<span style="font-weight: bold">)</span> to represent the biological brain.
&#10;    The BCNNs model is very similar to the neural networks utilized in deep learning, but has the added bonus of being able to <span style="color: #008000; text-decoration-color: #008000">'decouple'</span> the learning from the final results.
&#10;    BCNN for Machine Learning
&#10;    In order to make the transition from neural networks to BCNNs we will follow the same basic principles as we did with neural networks.
&#10;    However, instead of the neurons in neural networks that represent the data being represented, BCNNs work with nodes instead. This is because the nodes are the data, while the neurons are the information.
&#10;    In case you aren’t familiar with the term node, it is a symbol representing any type of data. For instance, it could be a datum in a neural network model.
&#10;    Another way to think of them is as symbols.
&#10;    The basic idea of nodes and connections is that a node can have many connections to other nodes, with each node linked to a connection to a larger entity.
&#10;    For instance, a node might have a target, which is just a point in space. A connection might have a value, which is just a number between <span style="color: #800080; text-decoration-color: #800080">0</span> and <span style="color: #800080; text-decoration-color: #800080">1</span>.
&#10;    Something like this:
&#10;    Node Value <span style="color: #800080; text-decoration-color: #800080">-0.1</span> <span style="color: #800080; text-decoration-color: #800080">0.1</span> <span style="color: #800080; text-decoration-color: #800080">0.1</span> <span style="color: #800080; text-decoration-color: #800080">0.1</span>
&#10;    The important thing to note, is that the value is a number between <span style="color: #800080; text-decoration-color: #800080">0</span> and <span style="color: #800080; text-decoration-color: #800080">1</span>.
&#10;    When we are given a list of data and an input, we will move forward through the data, connected nodes, and the resulting output.
&#10;    In the case of neural networks, this would look like:
&#10;    Neural Network
&#10;    A neural network is just a collection of nodes, connected to each other through connections.
&#10;    For example, let’s look at the ConvNet model from Wikipedia.
&#10;    Pretty simple. It has multiple layers of neurons, with each neuron being assigned one of the above variables.
&#10;    The neurons work with the data given as an input <span style="font-weight: bold">(</span>remember, it’s a
&#10;  </pre>

</div>

[^1]: in units of A100 `bfloat16` peak FLOPS
