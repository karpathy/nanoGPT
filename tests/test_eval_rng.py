"""Evaluating random validation batches must not alter the training sample order."""

import ast
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch


class _Model:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, x, y):
        assert not self.training
        return None, torch.tensor(1.0)


def _estimate_loss():
    train_file = Path(__file__).resolve().parents[1] / "train.py"
    tree = ast.parse(train_file.read_text(encoding="utf-8"), filename=str(train_file))
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "estimate_loss"
    )
    model = _Model()
    scope = {
        "torch": torch, "model": model, "eval_iters": 5,
        "get_batch": lambda split: (torch.randint(1000, (2, 3)), torch.ones(2, 3)),
        "ctx": nullcontext(),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(train_file), "exec"), scope)
    return scope["estimate_loss"], model


def test_validation_does_not_shift_the_next_training_batch_rng():
    estimate_loss, model = _estimate_loss()
    torch.manual_seed(321)
    expected = torch.randint(1000, (16,))

    torch.manual_seed(321)
    result = estimate_loss()
    actual = torch.randint(1000, (16,))

    assert torch.equal(actual, expected)
    assert result["train"] == pytest.approx(1.0)
    assert result["val"] == pytest.approx(1.0)
    assert model.training
