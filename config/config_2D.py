# Copyright (c) Meta Platforms, Inc. and affiliates
import time

# import tqdm
import torch
from dataclasses import dataclass

from torch.distributed.fsdp import StateDictType
from torch.utils.data import Dataset

from .base_config import base_config, fsdp_checkpointing_base, get_policy_base

# wrap model into FSDP container
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from model import CausalSelfAttention, MLP, GPT, GPTConfig, Block
from torch.distributed.fsdp import (
    ShardingStrategy,
)
import torch.distributed as dist


@dataclass
class train_config(base_config):
    # current models = "10.5M", "124M", "201M", "1B", "1.5B"
    model_name: str = "1B"
    use_tensor_parallel: bool = True

    dataset = "openwebtext"  # options = shakespeare_char, openwebtext
    data_dir = "data"

    iters_to_run: int = 8

    batch_size = 96
    block_size = 1024  # 256  # 1024 = gpt2, openwebtext, context of up to 256 previous characters
    use_bias: bool = False  # use bias in linear layers (recommend No)
    vocab_size: int = 50304  # use 65 for shakespeare, GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    dropout: float = 0.0

    # FSDP specific
    use_mixed_precision: bool = True
    wrapping_policy = ModuleWrapPolicy({CausalSelfAttention, MLP})
    model_sharding_strategy = ShardingStrategy.FULL_SHARD
    use_fsdp_activation_checkpointing: bool = True

    # stats - dynamic, not set by user
    current_model_params: int = 0


def set_mixed_precision_policy():
    from config.mixed_precision import get_mixed_precision_policy

    cfg = train_config()
    if cfg.use_mixed_precision:
        return get_mixed_precision_policy()
    else:
        return None


def build_model(cfg, tp_mesh=None, rank=None):
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
        # block_size: int = 1024
        # vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768

    elif model_name == "201M":
        n_layer: int = 16
        n_head: int = 16
        n_embd: int = 1024

    elif model_name == "1B":
        n_layer: int = 48
        n_head: int = 20
        n_embd: int = 1280

    elif model_name == "1.5B":
        n_layer: int = 46
        n_head: int = 20
        n_embd: int = 1600

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
    model = GPT(tp_mesh, gpt_conf, rank=rank)
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


def apply_checkpointing_policy(model):
    return fsdp_checkpointing_base(model, (MLP, CausalSelfAttention))
