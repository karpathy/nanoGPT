# Copyright (c) Meta Platforms, Inc. and affiliates
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    PairwiseParallel,
    parallelize_module,
)


def parallelize_model(model, model_config, twodim_mesh):
    for i in range(model_config.n_layer):
        block = model.get_submodule(f"transformer.h.{i}")
        parallelized_block = parallelize_module(
            module=block,
            device_mesh=twodim_mesh,
            parallelize_plan={
                "attn.c_attn": ColwiseParallel(),
                "attn.c_proj": RowwiseParallel(),
                "mlp": PairwiseParallel(),
            },
            tp_mesh_dim=1,
        )
        block = parallelized_block
    return i + 1
