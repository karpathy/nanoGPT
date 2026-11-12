"""Regression tests for train.py's standalone training-data sampler.

Importing train.py starts a training run, so extract the function from its
source and execute it with an isolated CPU-only namespace.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch


def _sampler(data_dir, batch_size, block_size):
    train_file = Path(__file__).resolve().parents[1] / "train.py"
    module = ast.parse(train_file.read_text(encoding="utf-8"), filename=str(train_file))
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_batch"
    )
    namespace = {
        "np": np,
        "torch": torch,
        "os": __import__("os"),
        "data_dir": str(data_dir),
        "batch_size": batch_size,
        "block_size": block_size,
        "device": "cpu",
        "device_type": "cpu",
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(train_file), "exec"), namespace)
    return namespace["get_batch"]


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("batch_size,block_size", [(1, 8), (12, 16), (32, 32)])
def test_get_batch_preserves_sampling_and_target_shift(tmp_path, split, batch_size, block_size):
    data = np.arange(257, dtype=np.uint16)
    (data if split == "train" else data[::-1]).tofile(tmp_path / f"{split}.bin")
    get_batch = _sampler(tmp_path, batch_size, block_size)

    torch.manual_seed(1234)
    indices = torch.randint(len(data) - block_size, (batch_size,))
    expected_data = data if split == "train" else data[::-1]
    expected_x = torch.tensor(
        np.stack([expected_data[int(i):int(i) + block_size] for i in indices]),
        dtype=torch.int64,
    )
    expected_y = torch.tensor(
        np.stack([expected_data[int(i) + 1:int(i) + block_size + 1] for i in indices]),
        dtype=torch.int64,
    )

    torch.manual_seed(1234)
    x, y = get_batch(split)
    assert torch.equal(x, expected_x)
    assert torch.equal(y, expected_y)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    assert x.is_contiguous() and y.is_contiguous()
    # model.GPT.forward flattens training targets using .view(-1).
    assert y.view(-1).numel() == batch_size * block_size


def test_get_batch_supports_single_valid_training_window(tmp_path):
    np.arange(17, dtype=np.uint16).tofile(tmp_path / "train.bin")
    get_batch = _sampler(tmp_path, batch_size=4, block_size=16)
    x, y = get_batch("train")
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])
