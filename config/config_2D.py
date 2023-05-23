import time

# import tqdm
import torch
from dataclasses import dataclass

from torch.distributed.fsdp import StateDictType
from torch.utils.data import Dataset

from .base_config import base_config, fsdp_checkpointing_base, get_policy_base

# wrap model into FSDP container
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from model import CausalSelfAttention, MLP, GPT
from torch.distributed.fsdp import (
    ShardingStrategy,
)
import torch.distributed as dist


@dataclass
class train_config(base_config):
    # current models = "10.5M", "124M"
    model_name: str = "10.5M"
    use_tensor_parallel: bool = True

    dataset = "shakespeare_char"
    data_dir = "data"

    iters_to_run: int = 21

    batch_size = 64
    block_size = 256  # context of up to 256 previous characters
    use_bias: bool = False  # use bias in linear layers (recommend No)
    vocab_size: int = 65  # 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    dropout: float = 0.0

    # FSDP specific
    wrapping_policy = ModuleWrapPolicy({CausalSelfAttention, MLP})
    model_sharding_strategy = ShardingStrategy.FULL_SHARD

    # stats - dynamic, not set by user
    current_model_params: int = 0


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


def build_model(cfg, tp_mesh=None):
    """load model config and return built model (from scratch)"""
    cfg = train_config()
    model_name = cfg.model_name

    if cfg.vocab_size is None:
        vocab_size = get_vocab_size(cfg)

    if model_name == "10.5M":
        # baby GPT model :)
        n_layer = 6
        n_head = 6
        n_embd = 384

    elif model_name == "124M":
        block_size: int = 1024
        vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768

    else:
        assert False, f"model {model_name} not supported yet."

    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=cfg.block_size,
        bias=cfg.use_bias,
        vocab_size=cfg.vocab_size,
        dropout=cfg.dropout,
    )

    gpt_conf = GPTConfig(**model_args)
    model = GPT(tp_mesh, gpt_conf)
    cfg.current_model_params = model.get_num_params()
    return model, gpt_conf


def get_vocab_size(cfg: train_config = None):
    import os
    import pickle

    if cfg is None:
        cfg = train_config()

    meta_path = os.path.join(cfg.data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    assert (
        meta_vocab_size is not None
    ), f"Failed to determine vocab size for {cfg.data_dir}"
    return meta_vocab_size
