# GPT-2 Medium
Sam Foreman
[<span class="orcid-green">{{< ai orcid >}}</span>](https://orcid.org/0000-0002-9981-0876)
2023-11-15

<span style="text-align:left;">[![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saforem2/nanoGPT/blob/master/notebooks/ngpt-gpt2-xl.ipynb)</span>

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

<pre style="white-space:pre;">
&#10;    /lus/grand/projects/datascience/foremans/locations/thetaGPU/projects/saforem2/nanoGPT/src/ngpt/__init__.py
    Has ngpt installed. Nothing to do.
&#10;</pre>

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

<pre style="white-space:pre;">
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
console.log(f'SEED: {SEED}')

rank = setup_torch('DDP', seed=1234)
cfg = get_config(
    [
        'data=owt',
        'model=gpt2_medium',
        'optimizer=gpt2_medium',
        'train=gpt2_medium',
        'train.dtype=bfloat16',
        'train.max_iters=1000',
        'train.log_interval=100',
        'train.init_from=gpt2-medium',
    ]
)
config = instantiate(cfg)
trainer = Trainer(config)
```

<div class="cell-output cell-output-display">

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:monospace,Menlo,'DejaVu Sans Mono',consolas,'Courier New'">
&#10;    --------------------------------------------------------------------------
    WARNING: There was an error initializing an OpenFabrics device.
&#10;      Local host:   thetagpu23
      Local device: mlx5_0
    --------------------------------------------------------------------------
    2023-11-15 09:48:20.191135: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
&#10;    <span style="color: #838383; text-decoration-color: #838383">[09:48:22] </span>SEED: <span style="color: #800080; text-decoration-color: #800080">627182480</span>
    <span style="color:#838383">[2023-11-15 09:48:23]</span><span style="color:var(--orange-text)">[WARNING]</span><span style="color:#777777">[configs.py:297</span><span style="color:#777777">]</span> - No meta.pkl found, assuming GPT-<span style="color:#A0A">2</span> encodings<span style="color:var(--orange-text)">...</span>
    <span style="color:#838383">[2023-11-15 09:48:26]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[configs.py:263</span><span style="color:#777777">]</span> - Rescaling GAS -> GAS <span style="color:#0A0">/</span><span style="color:#0A0">/</span> WORLD_SIZE = <span style="color:#A0A">1</span> <span style="color:#0A0">/</span><span style="color:#0A0">/</span> <span style="color:#A0A">1</span>
    <span style="color:#838383">[2023-11-15 09:48:26]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[configs.py:398</span><span style="color:#777777">]</span> - Tokens per iteration: <span style="color:#A0A">4</span>,<span style="color:#A0A">096</span>
    <span style="color:#838383">[2023-11-15 09:48:26]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[configs.py:430</span><span style="color:#777777">]</span> - Using <b><</b><b><span style="color:#F5F">torch.amp.autocast_mode.autocast</span></b><span style="color:#FFF"> object at </span><span style="color:#A0A">0x7fcbf3a11930</span><b>></b>
    <span style="color:#838383">[2023-11-15 09:48:26]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:187</span><span style="color:#777777">]</span> - Initializing from OpenAI GPT-<span style="color:#A0A">2</span> Weights: gpt2-medium
    [2023-11-15 09:48:32,281] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    <span style="color:#838383">[2023-11-15 09:48:49]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[model.py:225</span><span style="color:#777777">]</span> - loading weights from pretrained gpt: gpt2-medium
    <span style="color:#838383">[2023-11-15 09:48:49]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[model.py:234</span><span style="color:#777777">]</span> - forcing <i><span style="color:#55F">vocab_size</span></i>=<span style="color:#A0A">50257</span>, <i><span style="color:#55F">block_size</span></i>=<span style="color:#A0A">1024</span>, <i><span style="color:#55F">bias</span></i>=<i><span style="color:#5F5">True</span></i>
    <span style="color:#838383">[2023-11-15 09:48:49]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[model.py:240</span><span style="color:#777777">]</span> - overriding dropout rate to <span style="color:#A0A">0.0</span>
    <span style="color:#838383">[2023-11-15 09:48:55]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[model.py:160</span><span style="color:#777777">]</span> - number of parameters: <span style="color:#A0A">353.</span>77M
    Downloading (…)lve/main/config.json:   0%|| 0.00/718 [00:00<?, ?B/s]
    Downloading (…)"pytorch_model.bin";:   0%|| 0.00/1.52G [00:00<?, ?B/s]
    Downloading (…)neration_config.json:   0%|| 0.00/124 [00:00<?, ?B/s]
    <span style="color:#838383">[2023-11-15 09:49:16]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[model.py:290</span><span style="color:#777777">]</span> - num decayed parameter tensors: <span style="color:#A0A">98</span>, with <span style="color:#A0A">354</span>,<span style="color:#A0A">501</span>,<span style="color:#A0A">632</span> parameters
    <span style="color:#838383">[2023-11-15 09:49:16]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[model.py:291</span><span style="color:#777777">]</span> - num non-decayed parameter tensors: <span style="color:#A0A">194</span>, with <span style="color:#A0A">321</span>,<span style="color:#A0A">536</span> parameters
    <span style="color:#838383">[2023-11-15 09:49:17]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[model.py:297</span><span style="color:#777777">]</span> - using fused AdamW: <i><span style="color:#5F5">True</span></i>
&#10;</pre>

</div>

## Prompt (**prior** to training)

``` python
query = "What is a supercomputer? Explain like I'm a child, and speak clearly. Double check your logic.."
outputs = trainer.evaluate(query, num_samples=1, display=False)
console.print(fr'\[prompt]: "{query}"')
console.print("\[response]:\n\n" + fr"{outputs['0']['raw']}")
```

<div class="cell-output cell-output-display">

<pre style="white-space:pre;">
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer? Explain like I'm a child, and speak clearly. Double check your logic.."</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer? Explain like I'm a child, and speak clearly. Double check your logic.. When I ask you to describe your computer, this is what you know: a computer that runs a quantum computer to run a computer program. There are now two supercomputers that can do quantum computations, so how can you explain quantum computing? We will get into what quantum computing is first, and then we are going to talk to you about what is a supercomputer. You will learn how to explain quantum computing to a child, and you will find out that quantum computing is what is called artificial intelligence <span style="font-weight: bold">(</span>AI<span style="font-weight: bold">)</span>. Do you understand quantum computing?
    Follow up – Answering the Quantum Computing Questions
    Cognitive scientists are trying to define what is called artificial intelligence <span style="font-weight: bold">(</span>AI<span style="font-weight: bold">)</span>. They are trying to understand what computers can do. At the heart of the argument are the following three questions:
    What is AI and how does it apply to science?
    What if I say, <span style="color: #008000; text-decoration-color: #008000">"A computer can tell me if I am an elf, a lion, a fox, or a squirrel?"</span>. Can I make a decision?
    How does AI fit into the human condition?
    Have you ever thought about what is intelligence? Have you ever wondered, <span style="color: #008000; text-decoration-color: #008000">"What if I want to create a computer that can build me an AI, and that computer is smarter than me? How will the AI affect my life?"</span>.
    Before you even begin…
    BECAUSE YOU HAVE NOT BEEN PURCHASED
    This is where you may want to get an answer to a question that you have never had the opportunity to ask, and which is too big to ask right now. Think about it. Will I be able to make decisions? Will I be able to ask questions? Do I have the right to be an AI?
    With an intelligent AI, you can have the ability to run my brain. You can have your mind, and you can control it. You can control your body, you can control your thoughts, and you can control your actions. That will make you a living being. If you are able to experience the world like the rest of us, you will have a better understanding of what it means to be alive; a living being. You will have the ability to observe, dream, and experience. All this will be possible to you if you are created as an intelligent being.
    By creating an AI, this will be easier. It will be easy for someone to create an AI, and then run it as if
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

<pre style="white-space:pre;">
&#10;    <span style="color:#838383">[2023-11-15 09:50:50]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">100</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.791</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">387</span><span style="color:#A0A">.530</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.580</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.011</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">24</span><span style="color:#A0A">.642</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.837</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.826</span>
    <span style="color:#838383">[2023-11-15 09:51:28]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">200</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.716</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">375</span><span style="color:#A0A">.216</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.665</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.011</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">24</span><span style="color:#A0A">.722</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.837</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.826</span>
    <span style="color:#838383">[2023-11-15 09:52:07]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">300</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.701</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">398</span><span style="color:#A0A">.145</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.512</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.010</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">24</span><span style="color:#A0A">.649</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.837</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.826</span>
    <span style="color:#838383">[2023-11-15 09:52:45]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">400</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.858</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">376</span><span style="color:#A0A">.159</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.658</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.011</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">24</span><span style="color:#A0A">.722</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.837</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.826</span>
    <span style="color:#838383">[2023-11-15 09:53:24]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">500</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.542</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">422</span><span style="color:#A0A">.272</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.368</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.010</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">24</span><span style="color:#A0A">.512</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.837</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.826</span>
    <span style="color:#838383">[2023-11-15 09:54:03]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">600</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.912</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">406</span><span style="color:#A0A">.393</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.461</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.010</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">24</span><span style="color:#A0A">.410</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.837</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.826</span>
    <span style="color:#838383">[2023-11-15 09:54:42]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">700</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.862</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">369</span><span style="color:#A0A">.661</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.705</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.011</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">24</span><span style="color:#A0A">.552</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.837</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.826</span>
    <span style="color:#838383">[2023-11-15 09:55:21]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">800</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.849</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">336</span><span style="color:#A0A">.193</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.974</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.012</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">24</span><span style="color:#A0A">.938</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.837</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.826</span>
    <span style="color:#838383">[2023-11-15 09:56:00]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">900</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.586</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">387</span><span style="color:#A0A">.251</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.582</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.011</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">24</span><span style="color:#A0A">.910</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.837</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.826</span>
    <span style="color:#838383">[2023-11-15 09:56:38]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">1000</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.763</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">373</span><span style="color:#A0A">.859</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.675</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.011</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">24</span><span style="color:#A0A">.973</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.837</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.826</span>
&#10;</pre>

</div>

## Evaluate Model

``` python
query = "What is a supercomputer? Explain like I'm a child, and speak clearly. Double check your logic.."
outputs = trainer.evaluate(query, num_samples=1, display=False)
console.print(fr'\[prompt]: "{query}"')
console.print("\[response]:\n\n" + fr"{outputs['0']['raw']}")
```

<div class="cell-output cell-output-display">

<pre style="white-space:pre;">
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer? Explain like I'm a child, and speak clearly. Double check your logic.."</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer? Explain like I'm a child, and speak clearly. Double check your logic.. Does it say that when you say <span style="color: #008000; text-decoration-color: #008000">"I want to be famous"</span> that you should be famous as a songwriter too? Does it say that you should have a place in your life or in an organization? Is it saying that you should be regarded as a <span style="color: #008000; text-decoration-color: #008000">"supercomputer"</span> and have access to the best software on earth?
&#10;    Is it saying that you are a computer programmer?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer scientist"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer scientist"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;     does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;    does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer"</span>?<span style="color: #008000; text-decoration-color: #008000">" Doesn it say "</span>you are a computer programmer?"
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;    Does it say <span style="color: #008000; text-decoration-color: #008000">"you are a computer programmer?"</span>
&#10;    Does
&#10;</pre>

</div>

## Train a bit more??

``` python
trainer.model.module.train()
for iter in range(10):
    console.rule(f'iter: {iter}')
    trainer.train(train_iters=100)
    query = "What is a supercomputer?"
    outputs = trainer.evaluate(query, num_samples=1, display=False)
    console.print(fr'\[prompt]: "{query}"')
    console.print("\[response]:\n\n" + fr"{outputs['0']['raw']}")
    console.rule()
```

<div class="cell-output cell-output-display">

<pre style="white-space:pre;">
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────── </span>iter: <span style="color: #800080; text-decoration-color: #800080">0</span><span style="color: #00ff00; text-decoration-color: #00ff00"> ─────────────────────────────────────────────────────</span>
&#10;    <span style="color:#838383">[2023-11-15 09:58:35]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">1100</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.939</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">356</span><span style="color:#A0A">.182</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.808</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.011</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">26</span><span style="color:#A0A">.810</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span>
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer?
&#10;    A supercomputer consists of two parts: “a processor and a network of interconnected graphics processors, called supercomputers,” said Richard Andrews, a computer scientist at the University of Edinburgh.
&#10;    Image copyright PA Image caption The first three are the biggest computers in the world, but they’re all fitted with a touchscreen
&#10;    The processor is a small computer, usually a single-core one. Usually it is made of silicon and it’s the first, and most expensive, part of the complex system. The other two parts of the system are: a chip to power the processor and a memory and software stack to allow for the system to interact on the internet.
&#10;    The first is the processor. It runs a complex combination of algorithms. These algorithms work to compress higher-level data, but also do some processing to understand the lower levels of the system.
&#10;    According to the Oxford computer science textbook, algorithms are:
&#10;    It’s a special structure which allows a system to achieve a particular level of efficiency, such as a faster processor or a faster system of atoms or molecules.
&#10;    Researchers use algorithms to implement certain types of systems like computers or medicine. But a supercomputer can also work faster, or faster, to solve problems.
&#10;    Image copyright PA Image caption The supercomputer in the US can run things like social networks and video games
&#10;    A supercomputer has a processor inside, but there is also a memory and software to process it. This allows a system to do something faster. This is called a supercomputer’s memory.
&#10;    Image copyright Getty Images Image caption Computing power has increased, but it’s also improved the power of computers in recent years
&#10;    The second piece of the system is the computer network. This is a computer system that works together to perform calculations.
&#10;    This is a huge system. It is where all the computers in the world - computers in different countries, continents and different time zones - work together.
&#10;    The network works and converts the data on the computers into calculations. This is the core of a supercomputer.
&#10;    What goes on inside a supercomputer?
&#10;    Image copyright PA Image caption The supercomputer works together to work out human intelligence
&#10;    It can interact with the computer network via a touchscreen.
&#10;    The other pieces of the system are:
&#10;    A computer system can send and receive data through a touchscreen. It is
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────────────────────────────────────────────────────────────────────</span>
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────── </span>iter: <span style="color: #800080; text-decoration-color: #800080">1</span><span style="color: #00ff00; text-decoration-color: #00ff00"> ─────────────────────────────────────────────────────</span>
&#10;    <span style="color:#838383">[2023-11-15 09:59:28]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">1200</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.907</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">400</span><span style="color:#A0A">.617</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.496</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.010</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">23</span><span style="color:#A0A">.837</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span>
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer?
&#10;    One supercomputer is a super computer that is capable of the supercomputing of supercomputers and other machines.
&#10;    Where is the supercomputer on the circuit?
&#10;    The supercomputer is made up of several supercomputers.
&#10;    A supercomputer is the most advanced computer that has ever been built.
&#10;    It is so powerful that it can do so much work.
&#10;    It is the highest powered supercomputer.
&#10;    It is very powerful.
&#10;    It can do so much.
&#10;    It is capable of doing so much.
&#10;    It is supercomputer.
&#10;    It is supercomputer.
&#10;    It is supercomputer.
&#10;    It is supercomputer.
&#10;    It is supercomputer.
&#10;    Supercomputer.
&#10;    If you want a break-down of what a supercomputer is, read this.
&#10;    You might also want to read this.
&#10;    The diagram below shows that the supercomputer is connected to the circuit.
&#10;    What does that diagram look like?
&#10;    A pretty huge diagram, right? Well, it's a diagram that even the best supercomputer doesn't have. It is a diagram that only the best supercomputer has.
&#10;    And that is why it's called a diagram.
&#10;    It is called a diagram.
&#10;    A diagram is an electrical diagram.
&#10;    The diagram on the circuit has an electrical diagram of it's components.
&#10;    When you see an electrical diagram it means the diagram on the circuit has an electrical diagram of it's components.
&#10;    This diagram doesn't imply that the components on the circuit are connected.
&#10;    It just means that the diagram on the circuit has an electrical diagram of it's components.
&#10;    This diagram doesn't imply that the components on the circuit are connected.
&#10;    It just means that the diagram on the circuit has an electrical diagram of it's components.
&#10;    The diagram on the circuit has an electrical diagram of it's components.
&#10;    The diagram on the circuit has an electrical diagram of it's components.
&#10;    The diagram on the circuit has an electrical diagram of it's components.
&#10;    The diagram on the circuit has an electrical diagram of it's components.
&#10;    The diagram on the circuit has an electrical diagram of it's components.
&#10;    This diagram doesn't imply that the components on the circuit are connected.
&#10;    It just means that the diagram on the circuit has an electrical diagram of it
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────────────────────────────────────────────────────────────────────</span>
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────── </span>iter: <span style="color: #800080; text-decoration-color: #800080">2</span><span style="color: #00ff00; text-decoration-color: #00ff00"> ─────────────────────────────────────────────────────</span>
&#10;    <span style="color:#838383">[2023-11-15 10:00:22]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">1300</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.941</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">423</span><span style="color:#A0A">.726</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.360</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.010</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">22</span><span style="color:#A0A">.537</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span>
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer? Supercomputing machines are extremely powerful and cheap. They are also cheap to upgrade, and they are incredibly easy to install - and they can handle any challenge you throw at them. You can even rent out one for a few hours in a day <span style="font-weight: bold">(</span>after all, you need a supercomputer in the first place<span style="font-weight: bold">)</span>.
&#10;    Read more about supercomputing in our article on computing history.<span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">|endoftext|</span><span style="color: #000000; text-decoration-color: #000000">&gt;An estimated </span><span style="color: #800080; text-decoration-color: #800080">130</span><span style="color: #000000; text-decoration-color: #000000"> people were killed and at least </span><span style="color: #800080; text-decoration-color: #800080">200</span><span style="color: #000000; text-decoration-color: #000000"> wounded by militants in the attack.</span>
&#10;    <span style="color: #000000; text-decoration-color: #000000">The attack started near the scene of Friday's explosions, which killed two women and a child.</span>
&#10;    <span style="color: #000000; text-decoration-color: #000000">A bomber had targeted a residential area next to the scene of Friday's blasts.</span>
&#10;    <span style="color: #000000; text-decoration-color: #000000">The suicide bomber also targeted the building in the village of Bishnagar, which has a military installation.</span>
&#10;    <span style="color: #000000; text-decoration-color: #000000">There's no estimate of how many people were killed or wounded.</span>
&#10;    <span style="color: #000000; text-decoration-color: #000000">The blasts took place near the village of Bishnagar, which has a military installation. </span><span style="color: #000000; text-decoration-color: #000000; font-weight: bold">(</span><span style="color: #000000; text-decoration-color: #000000">Photo/AP</span><span style="color: #000000; text-decoration-color: #000000; font-weight: bold">)</span>
&#10;    <span style="color: #000000; text-decoration-color: #000000">One of the explosions took place near the building in the village of Bishnagar, which has a military installation. </span><span style="color: #000000; text-decoration-color: #000000; font-weight: bold">(</span><span style="color: #000000; text-decoration-color: #000000">Photo/AP</span><span style="color: #000000; text-decoration-color: #000000; font-weight: bold">)</span>
&#10;    <span style="color: #000000; text-decoration-color: #000000">The explosions took place near the building in the village of Bishnagar, which has a military installation. </span><span style="color: #000000; text-decoration-color: #000000; font-weight: bold">(</span><span style="color: #000000; text-decoration-color: #000000">Photo/AP</span><span style="color: #000000; text-decoration-color: #000000; font-weight: bold">)</span><span style="color: #000000; text-decoration-color: #000000">&lt;|endoftext|</span><span style="font-weight: bold">&gt;</span>How do you know what's happening on your computer? The days are long and the weeks are long, but once in a while, it's nice to get a peek inside the code behind what you’re doing. The difference between a good and a bad computer is usually the code behind its parts. It’s also a good sign that the developer is smart and is playing a skilled game of communication with a user. But you don’t have to play a game of communication to understand and understand how a code is actually interacting with the real world. If this sounds like a bit of a stretch, consider this: Google Chrome has evolved into a fast-paced browser focused on gaming and entertainment. Apple’s Safari was once full of the same stuff, but now it’s mostly silent.
&#10;    That’s a good thing, but it doesn’t explain why Apple’s mobile operating system is so focused on gaming. To the best of my knowledge, there’s not much meaningfully different about what the developer's thinking is as it relates to the development of Safari. Maybe it’s a good thing they�
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────────────────────────────────────────────────────────────────────</span>
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────── </span>iter: <span style="color: #800080; text-decoration-color: #800080">3</span><span style="color: #00ff00; text-decoration-color: #00ff00"> ─────────────────────────────────────────────────────</span>
&#10;    <span style="color:#838383">[2023-11-15 10:01:14]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">1400</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.948</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">366</span><span style="color:#A0A">.042</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.732</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.011</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">26</span><span style="color:#A0A">.088</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span>
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer?
&#10;    A supercomputer is an object that is built around a central processor.
&#10;    Although its size and weight may not be as strong as a human body, it is still far from the feeble components of a regular computer and is the same power as a small power station.
&#10;    The technology is called a supercomputer because it is capable of storing and processing huge amounts of data. If our supercomputer is not powerful enough to perform the calculations of a single human, it is equipped with the ability to run the complex software written in the instructions of the operating system.
&#10;    We have managed to get a number of super computers ready for testing, and they live up to their names.
&#10;    We are talking about super computers that take <span style="color: #800080; text-decoration-color: #800080">10</span>,<span style="color: #800080; text-decoration-color: #800080">000</span> times the power of the entire computer, and are equipped with a powerful combination of processors and big memory storage.
&#10;    They are capable of running the complex software written in the instructions of the operating system.
&#10;    You can call an IBM super computer a super computer because it has the power of all of a human being.
&#10;    How well does it work?
&#10;    IBM super computers are so powerful that they can run our own software. They are incredibly fast, are very long-lasting, they are able to communicate with each other and are connected with our WiFi network.
&#10;    The super computer of yours is a super computer that has a massive amount of memory and power, which is very much more powerful than a human average.
&#10;    That means it can run our programs, which are made of thousands of Java code.
&#10;    It is equipped with the power of a human at <span style="color: #800080; text-decoration-color: #800080">1</span>,<span style="color: #800080; text-decoration-color: #800080">500</span> times their humanweight.
&#10;    You can call an IBM super computer a super computer because it has the power of all of a human being.
&#10;    So, if you are a human being, and you have a super computer, will you be able to run a program that people might not have seen on a super computer?
&#10;    It is the same question as every other question.
&#10;    We have made a number of super computers for different purposes.
&#10;    We have shown to humans the power of running an ultra-fast super computer, such as a supercomputer of this kind.
&#10;    It is possible to run a program written in Java that is <span style="color: #800080; text-decoration-color: #800080">10</span>,<span style="color: #800080; text-decoration-color: #800080">000</span> times the power of the entire computer.
&#10;    We know that we have managed to get a number of super computers ready for testing.
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────────────────────────────────────────────────────────────────────</span>
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────── </span>iter: <span style="color: #800080; text-decoration-color: #800080">4</span><span style="color: #00ff00; text-decoration-color: #00ff00"> ─────────────────────────────────────────────────────</span>
&#10;    <span style="color:#838383">[2023-11-15 10:02:07]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">1500</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.871</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">341</span><span style="color:#A0A">.233</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.931</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.012</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">27</span><span style="color:#A0A">.985</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span>
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer?
&#10;    A supercomputer is a computer that is able to do complex calculations faster than a human can.
&#10;    A supercomputer is a computer that is able to do complex calculations faster than a human can.
&#10;    A supercomputer can have more cores than a human.
&#10;    A supercomputer is not a memory. It is a super computer. Once you have one, you can use it as an efficient computer.
&#10;    A supercomputer can be the fastest computer on the planet.
&#10;    <span style="font-weight: bold">(</span>Click through the gallery for a larger view.<span style="font-weight: bold">)</span>
&#10;    <span style="color: #008000; text-decoration-color: #008000">"We have high-performance supercomputers that are very fast and very efficient. The problem is that, because of modern computing technology, we can do multiple tasks at the same time,"</span> said Thomas DeLong, a computer scientist at the University of Michigan and a computer scientist at the University of California, Berkeley, who is one of the first members of the UNDP supercomputing committee.
&#10;    <span style="color: #008000; text-decoration-color: #008000">"This is not a supercomputer. It is a supercomputer. It is fast enough to do certain things that other computers cannot. That is the power of super computers."</span>
&#10;    DeLong's group has been working to upgrade super computers to get a better computing power. Their goal is to make the supercomputer that is needed for the next generation of computing on the planet -- the next generation that is computing the vast majority of the world's information -- the fastest supercomputer on the planet.
&#10;    <span style="color: #008000; text-decoration-color: #008000">"We have high-performance supercomputers that are very fast and very efficient. The problem is that, because of modern computing technology, we can do multiple tasks at the same time. That is the power of super computers,"</span> said Thomas DeLong, a computer scientist at the University of Michigan and a computer scientist at the University of California, Berkeley, who is one of the first members of the UNDP supercomputing committee. <span style="color: #008000; text-decoration-color: #008000">"That is what super computers are all about."</span>
&#10;    Dewey said that it's so easy to see why super computers could be faster than humans -- as an example.
&#10;    <span style="color: #008000; text-decoration-color: #008000">"Super computers are very fast computers. If you run a program that you're fast enough, you can live with it. It's not any longer a supercomputer, but a supercomputer, which is the next evolutionary step,"</span> said Dewey.
&#10;    Dewey said super computers are more powerful than the average human -- especially when there's more to computing.
&#10;    "The supercomputer
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────────────────────────────────────────────────────────────────────</span>
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────── </span>iter: <span style="color: #800080; text-decoration-color: #800080">5</span><span style="color: #00ff00; text-decoration-color: #00ff00"> ─────────────────────────────────────────────────────</span>
&#10;    <span style="color:#838383">[2023-11-15 10:03:00]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">1600</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.997</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">416</span><span style="color:#A0A">.252</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.402</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.010</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">22</span><span style="color:#A0A">.941</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span>
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer?
&#10;    A supercomputer is a computer for which a large number of computers are required to run programs. A supercomputer is a computer with an operating system that is modified to reduce the number of computers required to run programs.
&#10;    For example, a supercomputer is a computer with a Windows operating system that meets the standard of performance of a modern computer. This is roughly equivalent to a computer of similar size that runs an operating system that runs an operating system that doubles as a computer. This is the very computer that is needed to read and write programs and monitor them, and uses the internet to connect to a computer.
&#10;    Some examples of supercomputers are:
&#10;    Microsoft
&#10;    Google
&#10;    Apple
&#10;    Apple
&#10;    Another example of a computer with an operating system that also includes operating systems and applications that run on the internet is:
&#10;    Microsoft
&#10;    Google
&#10;    Apple
&#10;    Google
&#10;    Apple
&#10;    To communicate with a computer, a computer needs two sources of data:
&#10;    data directly from the command line
&#10;    data from libraries
&#10;    If you have a computer with more than two accounts on the command line, you want to take advantage of the power of the three-way communication paradigm developed by the World Wide Web. If you have any other system you're running <span style="font-weight: bold">(</span>e.g. a mobile device or laptop computer<span style="font-weight: bold">)</span>, you should see that the above is a more appropriate example of communication with a computer.
&#10;    Don't forget to get some free training on how to set up an email server on your computer, and learn how to use a command line terminal or graphical user interface on your computer. That way you can do all of this without having to install any new software on your computer.
&#10;    Or, if you are running a modern computer, you can download and install the free software that is needed for running an operating system on the command line from a single computer.
&#10;    When you're running a modern computer, you need to use software that is capable of running a modern operating system. If you are writing a program that uses the command line, you should read the content of the installation instructions to make sure that the correct software is installed. If the installation instructions don't mention the proper path for the software you are writing, you want to be sure you have the correct software installed. If you're using an older computer <span style="font-weight: bold">(</span>i.e. before the end of <span style="color: #800080; text-decoration-color: #800080">2010</span><span style="font-weight: bold">)</span>, you may need to install extra software that is
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────────────────────────────────────────────────────────────────────</span>
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────── </span>iter: <span style="color: #800080; text-decoration-color: #800080">6</span><span style="color: #00ff00; text-decoration-color: #00ff00"> ─────────────────────────────────────────────────────</span>
&#10;    <span style="color:#838383">[2023-11-15 10:03:53]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">1700</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">3</span><span style="color:#A0A">.223</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">392</span><span style="color:#A0A">.194</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.550</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.010</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">24</span><span style="color:#A0A">.349</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span>
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer?
&#10;    A supercomputer is computer that can run instructions that are cheaper to run than the typical computer.
&#10;    But how does the supercomputer perform?
&#10;    In the early days of computers, supercomputers were just very powerful computers, essentially.
&#10;    Then, in the early days of computers, supercomputers were just very powerful computers.
&#10;    Today, computers performed three services.
&#10;    It performed the simple tasks that we all perform.
&#10;    It could run and read programs.
&#10;    It could output an image.
&#10;    It could track specific objects, such as a physical object when it was moved and moved.
&#10;    It could create a software program.
&#10;    And it could create a program.
&#10;    At least that's what the supercomputer did.
&#10;    What's one of the most common supercomputers?
&#10;    You have a lot of supercomputers.
&#10;    That's because supercomputers are just massive computers.
&#10;    Different computers perform different tasks.
&#10;    What are the most common supercomputers?
&#10;    Computers are just computers.
&#10;    What are the most inexpensive supercomputers?
&#10;    Computers are everywhere.
&#10;    What's the cost to build a supercomputer that can do the most complicated tasks?
&#10;    The supercomputer is made out of three parts.
&#10;    It's made of silicon <span style="font-weight: bold">(</span>the most common material<span style="font-weight: bold">)</span>
&#10;    It's made of metal, the second most common
&#10;    The last one is probably a little hard to work with that is
&#10;    The first one is a lot based on silicon
&#10;    Its starting price is around $<span style="color: #800080; text-decoration-color: #800080">50</span>, not that expensive
&#10;    The second one is the most expensive
&#10;    What's the cost to build a supercomputer that can run more complicated software than today's computers?
&#10;    Computers are super computers.
&#10;    But what about the cost to build a supercomputer that can do more complex software than today's computers?
&#10;    The price is a lot
&#10;    What's the cost to build a supercomputer that can perform more complex tasks than today's computers?
&#10;    The cost is just over $<span style="color: #800080; text-decoration-color: #800080">100</span>
&#10;    What happens if you build it to work alone?
&#10;    When you build it to perform alone, you'll have a supercomputer
&#10;    What's the cost of building a supercomputer that can do more complex software than today's computers?
&#10;    The price of a supercomputer that can perform more complex software
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────────────────────────────────────────────────────────────────────</span>
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────── </span>iter: <span style="color: #800080; text-decoration-color: #800080">7</span><span style="color: #00ff00; text-decoration-color: #00ff00"> ─────────────────────────────────────────────────────</span>
&#10;    <span style="color:#838383">[2023-11-15 10:04:45]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">1800</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">3</span><span style="color:#A0A">.046</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">353</span><span style="color:#A0A">.265</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.831</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.012</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">27</span><span style="color:#A0A">.032</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span>
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer?
&#10;    A supercomputer is a computer that can control and control all of your work. It is essentially a computer that is able to do everything a computer can do. There are a few different names for a supercomputer <span style="font-weight: bold">(</span>or supercomputer<span style="font-weight: bold">)</span>. In this article, we will focus on one that is nearly ubiquitous, the supercomputer. We will look at the specific tools and software that can be used to create these machines.
&#10;    Defining a supercomputer
&#10;    Intelligent and flexible
&#10;    At its most basic, a supercomputer is an computer that is able to control and control all of your work, all of which happens in the real world. What you would call a computer is a computer that is able to go through your work and store what you have done. This is what most people think of as a computer, when they think of the word. That is not a computer, however, what you are really talking about is your computer.
&#10;    At the end of the day, a supercomputer is a computer that is able to run your work and store what you have done.
&#10;    The programming language used in a supercomputer is text based. You may have heard of a program like C programming language. In this article, I will spend a lot of time explaining how to use C programming language. I will also show you how to use Visual Studio <span style="color: #800080; text-decoration-color: #800080">2016</span> which is just as simple as using C programming language. I will also show you how to develop an MVP of your MVP, which is also what I am talking about in this article.
&#10;    Some examples of programs that might use C programming language include:
&#10;    Creating a supercomputer with C coding language
&#10;    Using C programming language
&#10;    Using C-based programming language
&#10;    Modifying a supercomputer that is able to control all of your work
&#10;    Modifying a supercomputer that is able to control all of your work
&#10;    Modifying a supercomputer that is able to control all of your work
&#10;    Modifying a supercomputer that is able to control all of your work
&#10;    Modifying a supercomputer that is able to control all of your work
&#10;    Modifying a supercomputer that is able to control all of your work
&#10;    Modifying an MVP of your MVP that is also able to control all of your work
&#10;    Modifying a MVP of your MVP that is able to control all of your work
&#10;    Modifying a MVP of your MVP that is able to control all
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────────────────────────────────────────────────────────────────────</span>
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────── </span>iter: <span style="color: #800080; text-decoration-color: #800080">8</span><span style="color: #00ff00; text-decoration-color: #00ff00"> ─────────────────────────────────────────────────────</span>
&#10;    <span style="color:#838383">[2023-11-15 10:05:37]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">1900</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">3</span><span style="color:#A0A">.108</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">413</span><span style="color:#A0A">.097</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.421</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.010</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">23</span><span style="color:#A0A">.116</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span>
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer?
&#10;    A supercomputer is not just a computer. It is an instrument that allows us to perform an algorithm. A supercomputer can be used by scientists, researchers, and computing experts. A supercomputer can be used by our lives to perform mathematical tasks such as addition and subtraction, while also being useful as a tool for fun.
&#10;    What's a supercomputer like?
&#10;    There is a very specific reason why super computers are super computers. Super computers are often referred to as chips, or chips with chips.
&#10;    What's a supercomputer like?
&#10;    Super computers are designed for deep learning. They can be used by any researcher who wants to understand how information works on a large scale. Although super computers often have a single chip, it has a huge number of chips.
&#10;    What's a supercomputer like?
&#10;    Super computers are designed to be used by the military. Scientists use super computers to detect and track enemy aircraft. A supercomputer can also be used to run the full search engine on a large server.
&#10;    What's a supercomputer like?
&#10;    Super computers are being used by academics, who are using the technology to develop tools for research. In particular, super computers have been used to check in on the behavior of the human brain.
&#10;    What's a supercomputer like?
&#10;    Super machines have been used to run a wide range of research, such as the Human Genome Project, where the focus is on understanding what happens in the brain.
&#10;    What's a supercomputer like?
&#10;    Superputers can be used by medical professionals to run an automated treatment program. A supercomputer could be used as a tool to find bugs in a patient's brain.
&#10;    What's a supercomputer like?
&#10;    Super computers are being used by astronomers to study the stars. A supercomputer can also be used as a tool to find bugs in a patient's brain.
&#10;    What's a supercomputer like?
&#10;    As the name of the video game, Supercomputer and Supercomputer. Supercomputer in the name of Super Computer and Supercomputer in the name of Supercomputer and Supercomputer.
&#10;    What's a supercomputer like?
&#10;    Superputers can be used by computer scientists to analyze the structure of stars. A supercomputer can also be used as a tool to find bugs in a patient's brain.
&#10;    What's a supercomputer like?
&#10;    Superputers can be used by computer
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────────────────────────────────────────────────────────────────────</span>
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────── </span>iter: <span style="color: #800080; text-decoration-color: #800080">9</span><span style="color: #00ff00; text-decoration-color: #00ff00"> ─────────────────────────────────────────────────────</span>
&#10;    <span style="color:#838383">[2023-11-15 10:06:30]</span><span style="color:var(--blue-text)">[INFO]</span><span style="color:#777777">[trainer.py:516</span><span style="color:#777777">]</span> - <i><span style="color:#55F">step</span></i>=<span style="color:#A0A">2000</span> <i><span style="color:#55F">loss</span></i>=<span style="color:#A0A">3</span><span style="color:#A0A">.133</span> <i><span style="color:#55F">dt</span></i>=<span style="color:#A0A">378</span><span style="color:#A0A">.236</span> <i><span style="color:#55F">sps</span></i>=<span style="color:#A0A">2</span><span style="color:#A0A">.644</span> <i><span style="color:#55F">mtps</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.011</span> <i><span style="color:#55F">mfu</span></i>=<span style="color:#A0A">25</span><span style="color:#A0A">.247</span> <i><span style="color:#55F">train_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span> <i><span style="color:#55F">val_loss</span></i>=<span style="color:#A0A">0</span><span style="color:#A0A">.000</span>
&#10;    <span style="font-weight: bold">[</span>prompt<span style="font-weight: bold">]</span>: <span style="color: #008000; text-decoration-color: #008000">"What is a supercomputer?"</span>
&#10;    <span style="font-weight: bold">[</span>response<span style="font-weight: bold">]</span>:
&#10;    What is a supercomputer?
&#10;    Supercomputing is the concept of processing and storing information such as programs and images into a computer.
&#10;    What kind of information can supercomputing access?
&#10;    When used in supercomputing, it can access the entire computer.
&#10;    What is the most powerful reason to allow supercomputing to access data?
&#10;    Supercomputing can access all the data that a computer can read.
&#10;    What are the most challenging aspects of supercomputing?
&#10;    Most supercomputing tasks require only a basic understanding of CPU instructions.
&#10;    What are the advantages of using supercomputing?
&#10;    It’s a fully accessible method, which makes it accessible and easy to use.<span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">|endoftext|</span><span style="font-weight: bold">&gt;</span>The Denver Broncos had a four-year contract with former Denver Broncos offensive coordinator Bill O’Brien out. The Broncos never hired Broncos offensive coordinator Bill O’Brien as Broncos offensive coordinator. O’Brien brought in Broncos offensive coordinator Mike Klis to be the Broncos offensive coordinator. Klis was a Broncos offensive coordinator for two seasons.
&#10;    The Broncos hired Broncos offensive coordinator Mike Klis to be the Broncos offensive coordinator. Denver fired Bill O’Brien as Broncos offensive coordinator.
&#10;     Denver had a four-year contract with Bill O’Brien out. The Broncos never hired Bill O’Brien as Broncos offensive coordinator. O’Brien brought in Broncos offensive coordinator Mike Klis to be the Broncos offensive coordinator.
&#10;    In <span style="color: #800080; text-decoration-color: #800080">2009</span>, the Broncos hired Mike Klis, who was the Broncos offensive coordinator for two seasons.
&#10;     Broncos coach Bill O’Brien, left, and Mike Klis, right, in the Broncos offense. In <span style="color: #800080; text-decoration-color: #800080">2009</span>, Broncos coach Bill O’Brien, left, and Mike Klis, right, in the Broncos offense.
&#10;    It’s a matter of fact that Bill O’Brien brought in Peyton Manning to be the Broncos offensive coordinator. Broncos offensive coordinator Mike Klis , left, and Mike Klis, right, in the Broncos offense.
&#10;    DENVER Broncos coach Bill O’Brien, left, and Mike Klis, right, were the Broncos offensive coordinators. Broncos coach Bill O’Brien, left, and Mike Klis, right.
&#10;    Denver coaches Mike Klis and Mike Klis and Broncos offensive coordinator Mike Klis.
&#10;    Denver coaches Mike Klis and Mike Klis, left. Denver coaches Mike Klis and Mike Klis
&#10;    <span style="color: #00ff00; text-decoration-color: #00ff00">───────────────────────────────────────────────────────────────────────────────────────────────────────────────────</span>
&#10;</pre>

</div>

[^1]: in units of A100 `bfloat16` peak FLOPS
