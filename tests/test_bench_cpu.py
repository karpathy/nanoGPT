"""CPU-only regression tests for the standalone benchmarking script."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch


def _bench_source():
    path = Path(__file__).resolve().parents[1] / "bench.py"
    return path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_real_data_batches_work_on_cpu_without_pinning(tmp_path, monkeypatch):
    path, tree = _bench_source()
    get_batch = next(
        nested
        for node in tree.body if isinstance(node, ast.If)
        for nested in node.body
        if isinstance(nested, ast.FunctionDef) and nested.name == "get_batch"
    )
    np.arange(32, dtype=np.uint16).tofile(tmp_path / "train.bin")
    training_data = np.memmap(tmp_path / "train.bin", dtype=np.uint16, mode="r")
    monkeypatch.setattr(
        torch.Tensor, "pin_memory",
        lambda self, *args, **kwargs: pytest.fail("CPU tensors must not be pinned"),
    )
    scope = {
        "torch": torch, "np": np, "train_data": training_data,
        "batch_size": 3, "block_size": 7,
        "device": "cpu", "device_type": "cpu",
    }
    exec(compile(ast.Module(body=[get_batch], type_ignores=[]), str(path), "exec"), scope)
    x, y = scope["get_batch"]("train")
    assert x.shape == y.shape == (3, 7)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_cuda_synchronization_occurs_only_under_cuda_guard():
    _, tree = _bench_source()
    parents = {}

    def visit(node, parent=None):
        parents[node] = parent
        for child in ast.iter_child_nodes(node):
            visit(child, node)

    visit(tree)
    sync_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "synchronize"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
    ]
    assert len(sync_calls) == 2
    for call in sync_calls:
        parent = parents[call]
        while parent is not None and not isinstance(parent, ast.If):
            parent = parents[parent]
        assert parent is not None
        assert isinstance(parent.test, ast.Compare)
        assert ast.unparse(parent.test) == "device_type == 'cuda'"
