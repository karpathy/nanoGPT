# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from torch.distributed.fsdp import (
    MixedPrecision,
)

import torch
import torch.distributed as dist
import torch.cuda.nccl as nccl
from distutils.version import LooseVersion


def get_mixed_precision_policy():
    bf16_ready = (
        torch.version.cuda
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    if bf16_ready:
        bfSixteen = MixedPrecision(
            param_dtype=torch.bfloat16,
            # Gradient communication precision.
            reduce_dtype=torch.bfloat16,
            # Buffer precision.
            buffer_dtype=torch.bfloat16,
        )
        return bfSixteen
    else:
        print(
            f"mixed precision = bfloat16, but this gpu or nccl version does not support native BF16\n"
        )

    return None  # None policy = fp32
