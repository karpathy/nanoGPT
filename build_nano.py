# from https://github.com/awgu/nanoGPT/blob/fsdp/tp_model.py

import math
import warnings
from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


from model import Block, CausalSelfAttention, GPT, MLP

from tp_common import BlockBase, GPTConfig, new_gelu

from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)


def _warn_dropout():
    """
    TODO: We are disabling dropout for now because:
    1. ``self.attn_dropout`` raises an assertion error like:
       "output spec does not match with output!"
    2. Distributed random ops have not been implemented yet and pose some
       trickiness. For replicated dropout, each rank must ensure the same
       random seed so that each rank's result preserves replicatedness. For
       sharded dropout, the result should match that of local non-distributed
       dropout. @wanchaol plans to add this support in the future.
    """
    warnings.warn(
        "Disabling dropout since tensor parallel dropout is not yet supported"
    )


def _replicate_tensor(x: Union[torch.Tensor, DTensor], mesh: DeviceMesh) -> DTensor:
    """Replicates ``x`` as a ``DTensor`` if not already."""
    # NOTE: If `x` is not already a replicated DTensor, then it must be the
    # same on all ranks for correctness.
    replicate = [Replicate()]
    return (
        x.redistribute(mesh, replicate)
        if isinstance(x, DTensor)
        else DTensor.from_local(x, mesh, replicate, run_check=False)
    )


def _replicate_tensor_to_local(x: DTensor, mesh: DeviceMesh) -> torch.Tensor:
    """
    Replicates ``x``, possibly triggering an all-reduce to sum partial results,
    and returns it as a local tensor.
    """
    return x.redistribute(device_mesh=mesh, placements=[Replicate()]).to_local()


class TPCausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, mesh: DeviceMesh) -> None:
        """
        Initializes a tensor parallel causal self-attention module. The QKV
        projections are column-wise sharded, and the output projection is
        row-wise sharded.
        To achieve the sharding, we *replace* the existing parameters with
        new ones backed by ``DTensor``.
        NOTE: Because ``nn.Linear`` computes xA^T + b, column-wise sharding is
        on the 0th dim, and row-wise sharding is on the 1st dim.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            "Expects n_embd to be a multiple of n_head but got "
            f"n_embd: {config.n_embd} n_head: {config.n_head}"
        )
        self.mesh = mesh
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # K, Q, V projections concatenated
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        # Column-wise shard the QKV projection's weight and bias
        self.c_attn.weight = nn.Parameter(
            distribute_tensor(self.c_attn.weight, mesh, [Shard(0)])
        )  # (n_embd, 3 * n_embd / mesh.size)^T
        self.c_attn.bias = nn.Parameter(
            distribute_tensor(self.c_attn.bias, mesh, [Shard(0)])
        )  # (3 * n_embd / mesh.size,)

        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        # Row-wise shard the output project's weight and replicate its bias
        self.c_proj.weight = nn.Parameter(
            distribute_tensor(self.c_proj.weight, mesh, [Shard(1)])
        )  # (n_embd / mesh.size, n_embd)^T
        self.c_proj.bias = nn.Parameter(
            distribute_tensor(self.c_proj.bias, mesh, [Replicate()])
        )  # (n_embd,)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Replicate the causal mask
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        causal_mask = DTensor.from_local(
            causal_mask, self.mesh, [Replicate()], run_check=False
        )
        self.register_buffer("bias", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes causal self-attention, meaning that attention is only applied
        to the left in the input sequence.
        """
        assert isinstance(
            x, (torch.Tensor, DTensor)
        ), f"Expects Tensor or DTensor but got {type(x)}"
        assert x.ndim == 3, f"Expects 3D input but got {x.shape}"
        (
            batch_size,
            block_size,
            n_embd,
        ) = x.size()
        assert (
            n_embd == self.n_embd
        ), f"Expects {self.n_embd} for rightmost dim but got {n_embd}"
        n_head = self.n_head

        x = _replicate_tensor(x, self.mesh)

        qkv = self.c_attn(x)  # (batch_size, block_size, 3 * n_embd / mesh.size)
        # (batch_size, block_size, n_embd / mesh.size) for each of Q, K, V
        q, k, v = qkv.split(n_embd, dim=2)
        # (batch_size, n_head, block_size, n_embd / n_head / mesh.size)
        q = q.view(batch_size, block_size, n_head, n_embd // n_head).transpose(1, 2)
        k = k.view(batch_size, block_size, n_head, n_embd // n_head).transpose(1, 2)
        v = v.view(batch_size, block_size, n_head, n_embd // n_head).transpose(1, 2)

        # (batch_size, n_head, block_size, n_embd / n_head / mesh.size)
        # * (batch_size, n_head, n_embd / n_head / mesh.size, block_size)
        # -> (batch_size, n_head, block_size, block_size)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(
            self.bias[:, :, :block_size, :block_size] == 0, float("-inf")
        )
        attn = F.softmax(attn, dim=-1)
        # attn = self.attn_dropout(attn)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, block_size, n_embd)
        y = self.c_proj(y)  # (batch_size, block_size, n_embd)
        y = _replicate_tensor_to_local(y, self.mesh)
        # y = self.resid_dropout(y)
        return y


def _init_qkv_params_from_tensors(
    src_weight: torch.Tensor,
    src_bias: torch.Tensor,
    dst_weight: DTensor,
    dst_bias: DTensor,
    mesh: DeviceMesh,
) -> Tuple[DTensor, DTensor]:
    """
    Initializes new ``DTensor`` weight and bias from ``src_weight`` and
    ``src_bias``, replacing ``dst_weight`` and ``dst_bias``. These should
    correspond to the QKV weight and bias (from ``c_attn``).
    NOTE: This is a *hack* to initialize ``self.c_attn`` 's parameters from
    pre-existing tensors ``weight`` and ``bias``, which is used for parity
    testing against a purely local version. In real training, the elements
    should converge to the same result even without this.
    To see the issue, consider the weight and suppose we have mesh size 2.
    Column-wise sharding the weight means rank 0 has all of Q's weight and half
    of K's weight and rank 1 has the other half of K's weight and all of V's
    weight. Thus, we have to reshuffle data so that each rank gets a shard of
    each of Q, K, and V's weight. The same applies for the bias.
    Incorrect (naively):
    ________rank0________|_______rank1________
    |      |      |      |      |      |      |
    | W_Q0 | W_Q1 | W_K0 | W_K1 | W_V0 | W_V1 |
    |______|______|______|______|______|______|
    Correct (this method):
    ________rank0________|_______rank1________
    |      |      |      |      |      |      |
    | W_Q0 | W_K0 | W_V0 | W_Q1 | W_K1 | W_V1 |
    |______|______|______|______|______|______|
    """
    rank = mesh.get_rank()
    size = mesh.size()
    # (n_embd, 3 * n_embd)^T
    assert (
        dst_weight.shape == src_weight.shape
    ), f"Expects {dst_weight.shape} but got {src_weight.shape}"
    n_embd = dst_weight.shape[-1]
    assert n_embd % size == 0, f"n_embd: {n_embd} size: {size}"
    # (n_embd, n_embd)^T each
    weight_q, weight_k, weight_v = src_weight.split(n_embd, dim=0)
    # (n_embd, n_embd // mesh.size)^T each
    weight_q_shard = weight_q.split(n_embd // size, dim=0)[rank]
    weight_k_shard = weight_k.split(n_embd // size, dim=0)[rank]
    weight_v_shard = weight_v.split(n_embd // size, dim=0)[rank]
    # (n_embd, 3 * n_embd // mesh.size)^T
    weight_shard = torch.cat((weight_q_shard, weight_k_shard, weight_v_shard), dim=0)
    assert (
        dst_weight._local_tensor.shape == weight_shard.shape
    ), f"Expects {dst_weight._local_tensor.shape} but got {weight_shard.shape}"
    c_attn_weight = nn.Parameter(
        DTensor.from_local(
            weight_shard,
            dst_weight.device_mesh,
            dst_weight.placements,
            run_check=False,
        )
    )
    # (3 * n_embd,)
    assert (
        dst_bias.shape == src_bias.shape
    ), f"Expects {dst_bias.shape} but got {src_bias.shape}"
    # (n_embd,) each
    bias_q, bias_k, bias_v = src_bias.split(n_embd, dim=0)
    # (n_embd // mesh.size,) each
    bias_q_shard = bias_q.split(n_embd // size, dim=0)[rank]
    bias_k_shard = bias_k.split(n_embd // size, dim=0)[rank]
    bias_v_shard = bias_v.split(n_embd // size, dim=0)[rank]
    # (3 * n_embd // mesh.size,)
    bias_shard = torch.cat((bias_q_shard, bias_k_shard, bias_v_shard), dim=0)
    assert (
        dst_bias._local_tensor.shape == bias_shard.shape
    ), f"Expects {dst_bias._local_shard.shape} but got {bias_shard.shape}"
    c_attn_bias = nn.Parameter(
        DTensor.from_local(
            bias_shard,
            dst_bias.device_mesh,
            dst_bias.placements,
            run_check=False,
        )
    )
    return c_attn_weight, c_attn_bias


class TP_MLP(nn.Module):
    def __init__(self, config: GPTConfig, mesh: DeviceMesh) -> None:
        super().__init__()
        self.mesh = mesh

        # Column-wise shard the first linear's weight and bias
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_fc.weight = nn.Parameter(
            distribute_tensor(self.c_fc.weight, mesh, [Shard(0)])
        )  # (n_embd, 4 * n_embd / mesh.size)
        self.c_fc.bias = nn.Parameter(
            distribute_tensor(self.c_fc.bias, mesh, [Shard(0)])
        )  # (4 * n_embd / mesh.size,)

        # Row-wise shard the second linear's weight and replicate its bias
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.weight = nn.Parameter(
            distribute_tensor(self.c_proj.weight, mesh, [Shard(1)])
        )  # (4 * n_embd / mesh.size, n_embd)
        self.c_proj.bias = nn.Parameter(
            distribute_tensor(self.c_proj.bias, mesh, [Replicate()])
        )  # (n_embd,)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _replicate_tensor(x, self.mesh)
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        # x = self.dropout(x)
        x = _replicate_tensor_to_local(x, self.mesh)
        return x


class TPBlock(BlockBase):
    def __init__(self, config: GPTConfig, mesh: DeviceMesh) -> None:
        super().__init__(
            config,
            partial(TPCausalSelfAttention, mesh=mesh),
            partial(TP_MLP, mesh=mesh),
        )
        self.mesh = mesh


def parallelize_causal_self_attention(
    module: CausalSelfAttention,
    mesh: DeviceMesh,
) -> CausalSelfAttention:
    """
    Applies tensor parallelism to the causal self-attention module ``module``
    by column-wise sharding the QKV weight and bias and by row-wise sharding
    the output projection weight.
    """
    # Column-wise shard the QKV projection's weight and bias
    module.c_attn.weight = nn.Parameter(
        distribute_tensor(
            module.c_attn.weight, mesh, [Shard(0) for _ in range(mesh.ndim)]
        )
    )  # (n_embd, 3 * n_embd / mesh.size)^T
    # print(f"{module.c_attn.bias=}")
    if module.c_attn.bias:
        module.c_attn.bias = nn.Parameter(
            distribute_tensor(
                module.c_attn.bias, mesh, [Shard(0) for _ in range(mesh.ndim)]
            )
        )  # (3 * n_embd / mesh.size,)

    # Row-wise shard the output projection's weight and replicate its bias
    module.c_proj.weight = nn.Parameter(
        distribute_tensor(
            module.c_proj.weight, mesh, [Shard(1) for _ in range(mesh.ndim)]
        )
    )  # (n_embd / mesh.size, n_embd)^T
    if module.c_proj.bias:
        module.c_proj.bias = nn.Parameter(
            distribute_tensor(
                module.c_proj.bias, mesh, [Replicate() for _ in range(mesh.ndim)]
            )
        )  # (n_embd,)

    # Replicate the causal mask
    if module.c_attn.bias:
        module.bias = DTensor.from_local(
            module.bias, mesh, [Replicate()], run_check=False
        )

    if module.attn_dropout.p > 0 or module.resid_dropout.p > 0:
        _warn_dropout()
    module.attn_dropout = nn.Identity()
    module.resid_dropout = nn.Identity()

    module.register_forward_pre_hook(lambda _, x: _replicate_tensor(*x, mesh))
    module.register_forward_hook(
        lambda mod, inp, x: _replicate_tensor_to_local(x, mesh)
    )
    return module


def parallelize_mlp(
    module: MLP,
    mesh: DeviceMesh,
) -> MLP:
    """
    Applies tensor parallelism to the MLP module ``module`` by column-wise
    sharding the first linear's weight and bias and by row-wise sharding the
    second linear's weight.
    """
    # Column-wise shard the first linear's weight and bias
    module.c_fc.weight = nn.Parameter(
        distribute_tensor(
            module.c_fc.weight, mesh, [Shard(0) for _ in range(mesh.ndim)]
        )
    )  # (n_embd, 4 * n_embd / mesh.size)^T
    module.c_fc.bias = nn.Parameter(
        distribute_tensor(module.c_fc.bias, mesh, [Shard(0) for _ in range(mesh.ndim)])
    )  # (4 * n_embd / mesh.size)

    # Row-wise shard the second linear's weight and replicate its bias
    module.c_proj.weight = nn.Parameter(
        distribute_tensor(
            module.c_proj.weight, mesh, [Shard(1) for _ in range(mesh.ndim)]
        )
    )  # (4 * n_embd / mesh.size, n_embd)^T
    module.c_proj.bias = nn.Parameter(
        distribute_tensor(
            module.c_proj.bias, mesh, [Replicate() for _ in range(mesh.ndim)]
        )
    )  # (n_embd,)

    if module.dropout.p > 0:
        _warn_dropout()
    module.dropout = nn.Identity()

    module.register_forward_pre_hook(lambda _, x: _replicate_tensor(*x, mesh))
    module.register_forward_hook(
        lambda mod, inp, x: _replicate_tensor_to_local(x, mesh)
    )
    return module


def parallelize_block(
    module: Block,
    mesh: DeviceMesh,
) -> Block:
    """
    Applies tensor parallelism to the causal self-attention and MLP modules in
    the block. The layer norms are not parallelized, meaning that they are
    computed purely locally.
    """

    module.attn = parallelize_causal_self_attention(module.attn, mesh)
    print(f"===> Success - module attn parallelized")
    module.mlp = parallelize_mlp(module.mlp, mesh)
    return module


def parallelize_gpt(
    module: GPT,
    mesh: DeviceMesh,
) -> GPT:
    module.transformer["h"] = nn.ModuleList(
        [parallelize_block(block, mesh) for block in module.transformer["h"]]
    )
    return module
