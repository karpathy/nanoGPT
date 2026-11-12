"""Benchmark synchronization must target the device that runs the model."""

import ast
from pathlib import Path


def test_benchmark_cuda_sync_uses_selected_device():
    path = Path(__file__).resolve().parents[1] / "bench.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "synchronize"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
    ]
    assert len(calls) == 2
    assert all(len(call.args) == 1 for call in calls)
    assert all(
        isinstance(call.args[0], ast.Name) and call.args[0].id == "device"
        for call in calls
    )
