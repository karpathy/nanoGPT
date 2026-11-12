"""Avoid allocating full checkpoint tensors on GPU before model transfer."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["train.py", "sample.py"])
def test_checkpoint_loads_on_cpu_before_model_transfer(script):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / script).read_text(encoding="utf-8"))
    loads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    ]
    assert loads
    checkpoint_load = next(
        node for node in loads
        if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "ckpt_path"
    )
    assert any(
        kw.arg == "map_location"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value == "cpu"
        for kw in checkpoint_load.keywords
    )
