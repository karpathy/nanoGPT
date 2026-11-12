"""Regression tests for rank-zero-only validation in train.py.

Importing train.py would start a training run; isolate estimate_loss via AST.
"""

import ast
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch


class _EvalModel:
    def __init__(self):
        self.training = True
        self.calls = 0

    def __call__(self, x, y):
        assert not self.training
        self.calls += 1
        return None, torch.tensor(float(self.calls))

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _DDPStub:
    def __init__(self, module):
        self.module = module

    def __call__(self, *args):
        raise AssertionError("master-only evaluation must not enter DDP.forward")


@pytest.mark.parametrize("ddp", [False, True])
def test_master_only_evaluation_uses_unwrapped_module(ddp):
    train_file = Path(__file__).resolve().parents[1] / "train.py"
    module = ast.parse(train_file.read_text(encoding="utf-8"), filename=str(train_file))
    fn = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "estimate_loss"
    )
    unwrapped = _EvalModel()
    model = _DDPStub(unwrapped) if ddp else unwrapped
    environment = {
        "torch": torch,
        "ddp": ddp,
        "model": model,
        "eval_iters": 2,
        "get_batch": lambda split: (torch.ones(1, 2), torch.ones(1, 2)),
        "ctx": nullcontext(),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(train_file), "exec"), environment)
    result = environment["estimate_loss"]()
    assert set(result) == {"train", "val"}
    assert result["train"] == pytest.approx(1.5)
    assert result["val"] == pytest.approx(3.5)
    assert unwrapped.calls == 4
    assert unwrapped.training
