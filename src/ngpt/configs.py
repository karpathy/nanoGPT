"""
nanoGPT/config.py
"""
from __future__ import absolute_import, annotations, division, print_function
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, asdict
# import logging
import json
import os
from pathlib import Path
import pickle
from typing import Optional, Any
from copy import deepcopy
import tiktoken

from enrich import get_logger
# from ngpt import get_logger
from ezpz import get_rank, get_world_size
import numpy as np
import rich.repr
import torch
from hydra.core.config_store import ConfigStore
# from transformers import data

log = get_logger(__name__, 'INFO')
# log = logging.getLogger(__name__)

RANK = get_rank()
WORLD_SIZE = get_world_size()
if RANK == 0:
    log.setLevel("INFO")
else:
    log.setLevel("CRITICAL")


# -- Configure useful Paths -----------------------
# warnings.filterwarnings('ignore')
HERE = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = HERE.parent.parent
PROJECT_ROOT = PROJECT_DIR
CONF_DIR = HERE.joinpath('conf')
CACHE_DIR = PROJECT_DIR.joinpath('.cache')
WB_CACHE_DIR = CACHE_DIR.joinpath('wandb')
HF_DATASETS_CACHE_DIR = CACHE_DIR.joinpath('huggingface', 'datasets')
DATA_DIR = PROJECT_DIR.joinpath('data')
CKPT_DIR = HERE.joinpath('ckpts')
DS_CONFIG_PATH = CONF_DIR.joinpath('ds_config.yaml')
LOGS_DIR = HERE.joinpath('logs')
OUTPUTS_DIR = HERE.joinpath('outputs')

OUTDIRS_FILE = OUTPUTS_DIR.joinpath('outdirs.log')
CKPTS_FILE = CKPT_DIR.joinpath('checkpoints.log')

CKPT_DIR.mkdir(exist_ok=True, parents=True)
CONF_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
WB_CACHE_DIR.mkdir(exist_ok=True, parents=True)
HF_DATASETS_CACHE_DIR.mkdir(exist_ok=True, parents=True)

PT_DTYPES = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}

os.environ['WANDB_CACHE_DIR'] = WB_CACHE_DIR.as_posix()
HF_DATASETS_CACHE = os.environ.get('HF_DATASETS_CACHE', None)
if HF_DATASETS_CACHE is None:
    log.info(f'Setting HF_DATASETS_CACHE to {HF_DATASETS_CACHE_DIR.as_posix()}')
    os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE_DIR.as_posix()
else:
    log.info(f'Caught {HF_DATASETS_CACHE.as_posix()=} from env')

# os.environ['HF_DATASETS_CACHE'] = HF_DATASETS_CACHE_DIR.as_posix()



FRAMEWORKS = {
    'pytorch': ['p', 'pt', 'torch', 'pytorch'],
    'tensorflow': ['t', 'tf', 'tflow', 'tensorflow'],
}
BACKENDS = {
    'pytorch': ['ddp', 'ds', 'dspeed', 'deepspeed', 'h', 'hvd', 'horovod'],
    'tensorflow': ['h', 'hvd', 'horovod']
}

# @dataclass
# class GPTConfig:
#     block_size: int = 1024
#     vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
#     n_layer: int = 12
#     n_head: int = 12
#     n_embd: int = 768
#     dropout: float = 0.0
#     bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

def is_interactive() -> bool:
    from IPython.core.getipython import get_ipython
    # from IPython import get_ipython
    eval = os.environ.get('INTERACTIVE', None) is not None
    bval = get_ipython() is not None
    return (eval or bval)


def build_experiment(overrides: Optional[str | list[str]] = None):
    import warnings
    warnings.filterwarnings('ignore')
    from l2hmc.configs import get_config
    if isinstance(overrides, str):
        overrides = [overrides]
    cfg = get_config(overrides)
    return get_experiment(cfg=cfg)


def dict_to_list_of_overrides(d: dict):
    return [f'{k}={v}' for k, v in flatten_dict(d, sep='.').items()]


def flatten_dict(d: dict, sep: str = '/', pre='') -> dict:
    return {
        pre + sep + k if pre else k: v
        for kk, vv in d.items()
        for k, v in flatten_dict(vv, sep, kk).items()
    } if isinstance(d, dict) else {pre: d}


def add_to_outdirs_file(outdir: os.PathLike):
    with open(OUTDIRS_FILE, 'a') as f:
        f.write(Path(outdir).resolve().as_posix() + '\n')


def add_to_ckpts_file(outdir: os.PathLike):
    log.info(f'Appending {outdir} to {CKPTS_FILE}')
    with open(CKPTS_FILE, 'a') as f:
        f.write(Path(outdir).resolve().as_posix() + '\n')



def get_config(overrides: Optional[list[str]] = None):
    from hydra import (
        initialize_config_dir,
        compose
    )
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    overrides = [] if overrides is None else overrides
    with initialize_config_dir(
            CONF_DIR.absolute().as_posix(),
            version_base=None,
    ):
        cfg = compose('config', overrides=overrides)

    return cfg


@dataclass
@rich.repr.auto
class BaseConfig(ABC):

    @abstractmethod
    def to_str(self) -> str:
        pass

    def to_json(self) -> str:
        return json.dumps(
            {
                k: v.as_posix() if isinstance(v, Path) else v
                for k, v in asdict(self).items()
            },
            indent=4,
        )
        # sane_dict = {
        #     k: (
        #         v.as_posix() if isinstance(v, Path)
        #             else (asdict(v) if isinstance() else v)
        #
        #         # if not isinstance(v, Path) else v.as_posix()
        #
        #     )
        #     for k, v in self.__dict__.items()
        # }
        # return json.dumps(sane_dict)

    def get_config(self) -> dict:
        return asdict(self)

    def asdict(self) -> dict:
        return asdict(self)

    def to_dict(self) -> dict:
        return deepcopy(self.__dict__)

    def to_file(self, fpath: os.PathLike) -> None:
        with open(fpath, 'w') as f:
            json.dump(self.to_json(), f, indent=4)

    def from_file(self, fpath: os.PathLike) -> None:
        with open(fpath, 'w') as f:
            with open(fpath, 'r') as f:
                config = json.load(f)

        self.__init__(**config)

    def __getitem__(self, key):
        return super().__getattribute__(key)

@dataclass
class ModelConfig(BaseConfig):
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    batch_size: int = 12
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    vocab_size: Optional[int] = None

    def set_vocab_size(self, vocab_size: int):
        log.info(f'Resetting vocab_size from: {self.vocab_size} to {vocab_size}')
        self.vocab_size = vocab_size

    def set_block_size(self, block_size: int):
        log.info(f'Resetting block size from {self.block_size} to {block_size}')
        self.block_size = block_size

    def to_str(self):
        strs = [
            f'nL-{self.n_layer}',
            f'nH-{self.n_head}',
            f'nE-{self.n_embd}',
            f'mbs-{self.batch_size}',
            f'blk-{self.block_size}',
        ]
        if self.dropout > 0.:
            strs.append(f'dp-{self.dropout:.2f}'.replace('.', 'p'))
        if self.bias:
            strs.append('bias')
        return '_'.join(strs)


@dataclass
class OptimizerConfig(BaseConfig):
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    gradient_accumulation_steps: Optional[int] = None

    def to_str(self):
        strs = [
            f'lr-{self.learning_rate:.2f}'.replace('.', 'p'),
            f'b1-{self.beta1:.2f}'.replace('.', 'p'),
            f'b2-{self.beta2:.2f}'.replace('.', 'p'),
            f'clip-{self.grad_clip:.2f}'.replace('.', 'p'),
            f'gas-{self.gradient_accumulation_steps}',
        ]
        return '_'.join(strs)

    def __post_init__(self):
        if self.gradient_accumulation_steps is None or self.gradient_accumulation_steps == 1:
            self.gradient_accumulation_steps = WORLD_SIZE
        assert self.gradient_accumulation_steps % WORLD_SIZE == 0
        log.info(
            f"Rescaling GAS -> GAS // WORLD_SIZE "
            f"= {self.gradient_accumulation_steps} // {WORLD_SIZE}"
        )
        self.gradient_accumulation_steps //= WORLD_SIZE
        self.gas = self.gradient_accumulation_steps
        pass


@dataclass
class DataConfig(BaseConfig):
    out_dir: str = 'out'
    dataset: str = 'openwebtext'
    root_path: Optional[Any] = None

    def to_str(self):
        return f'dset-{self.dataset}'

    def __post_init__(self):
        self._root_path = (
            DATA_DIR if self.root_path is None
            else Path(self.root_dir)
        )
        self.data_dir = self._root_path.joinpath(self.dataset)
        self.ckpt_dir = CKPT_DIR.joinpath(self.out_dir)
        self.meta_path = self.data_dir.joinpath('meta.pkl')
        self.meta_vocab_size = None
        self._load_meta = self.meta_path.is_file()
        self.stoi = None
        self.itos = None
        if self._load_meta:
            with self.meta_path.open('rb') as f:
                meta = pickle.load(f)
            self.meta_vocab_size = meta['vocab_size']
            self._stoi = meta['stoi']
            self._itos = meta['itos']
            self.encode = lambda s: [self._stoi[c] for c in s]
            self.decode = lambda s: ''.join([self._itos[i] for i in s])
        else:
            log.warning('No meta.pkl found, assuming GPT-2 encodings...')
            self._enc = tiktoken.get_encoding('gpt2')
            self.encode = lambda s: self._enc.encode(s, allowed_special={'<|endoftext|>'})
            self.decode = lambda z: self._enc.decode(z)


@dataclass
class SampleConfig(BaseConfig):
    pass


@dataclass
class TrainConfig(BaseConfig):
    # model: ModelConfig
    # optimizer: OptimizerConfig
    # data: DataConfig
    framework: str = "pytorch"
    backend: str = "DDP"
    device: Optional[str] = None
    seed: Optional[int] = None
    port: Optional[str] = None
    # ds_config_path: Optional[os.PathLike] = None
    # wandb_project_name: Optional[str] = None
    precision: Optional[str] = None
    ngpus: Optional[int] = None
    use_wandb: bool = True
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'
    # wandb_log: bool = True
    wandb_project: str = 'nanoGPT'
    # wandb_run_name: str = 'gpt2'
    max_iters: int = 600000
    warmup_iters: int = 2000
    # backend: str = 'nccl'
    dtype: str = 'bfloat16'
    compile: bool = True

    def to_str(self):
        return '_'.join([
            self.wandb_project,
            # self.wandb_run_name,
            # f'dset-{self.dataset}',
            f'dtype-{self.dtype}',
            f'init-{self.init_from}',
        ])


@dataclass
class ExperimentConfig(BaseConfig):
    train: TrainConfig
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig

    def to_str(self):
        return '_'.join([
            self.train.to_str(),
            self.model.to_str(),
            self.data.to_str(),
            self.optimizer.to_str(),
        ])

    def reset_model_config(self, model_config: ModelConfig):
        self.model = model_config

    def reset_optimizer_config(self, optimizer_config: OptimizerConfig):
        self.optimizer = optimizer_config

    def set_iter_num(self, iter_num: int):
        log.info(f'Resetting iter_num from: {self.iter_num} to {iter_num}')
        self.iter_num = iter_num

    def set_best_val_loss(self, best_val_loss: float):
        log.info(
            f'Resetting best_val_loss from: '
            f'{self.best_val_loss} to {best_val_loss}'
        )
        self.best_val_loss = best_val_loss

    # def build_model(self, model_config: ModelConfig) -> torch.nn.Module:
    #         self.model = GPT(model_config)
    #         # gptconf = GPTConfig()
    #
    def __post_init__(self):
        self.rank = RANK
        self.world_size = get_world_size()
        self.main_process = (self.rank == 0)
        self.tokens_per_iter = (
            self.optimizer.gradient_accumulation_steps
            * WORLD_SIZE
            * self.model.batch_size
            * self.model.block_size
        )
        self.samples_per_iter = (
            self.optimizer.gradient_accumulation_steps
            * WORLD_SIZE
        )
        log.info(f'Tokens per iteration: {self.tokens_per_iter:,}')
        self.iter_num = 0
        self.best_val_loss = 1e9
        if self.model.vocab_size is None:
            self.model.vocab_size = (
                self.data.meta_vocab_size
                if self.data.meta_vocab_size is not None
                else 50304
            )
        self.train_data = np.memmap(
            self.data.data_dir.joinpath('train.bin'),
            dtype=np.uint16,
            mode='r'
        )
        self.val_data = np.memmap(
            self.data.data_dir.joinpath('val.bin'),
            dtype=np.uint16,
            mode='r'
        )
        self._out_dir = Path(self.data.out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ptdtype = PT_DTYPES[self.train.dtype]
        # self.device_type == 'cpu'
        self.ctx = (
            nullcontext() if not torch.cuda.is_available()
            else torch.autocast(  # type:ignore
                dtype=self.ptdtype,
                # enabled=self.enable_autocast,
                device_type=self.device_type,
            )
        )
        log.info(f'Using {self.ctx}')
        # else torch.amp.autocast(
        #     device_type=self.device_type,
        #     dtype=self.ptdtype
        # )
        if self.train.init_from == 'scratch':
            log.info('Initializing a new model from scratch')
            if self.data.meta_vocab_size is None:
                log.info(
                    'Defaulting to vocab_size of GPT-2 to 50304 '
                    '(50257 rounded up for efficiency)'
                )
            self.vocab_size = (
                self.data.meta_vocab_size 
                if self.data.meta_vocab_size is not None 
                else 50304
            )
        #     self.model = GPT(self.model_config)


cs = ConfigStore.instance()
cs.store(name='experiment_config', node=ExperimentConfig)
cs.store(name='train_config', node=TrainConfig)
cs.store(name='model_config', node=ModelConfig)
cs.store(name='data_config', node=DataConfig)
cs.store(name='optimizer_config', node=OptimizerConfig)
# cs.store(name='train_config', node=TrainConfig)
