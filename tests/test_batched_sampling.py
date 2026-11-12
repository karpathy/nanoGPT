"""Verify that sample.py can generate multiple independent samples per forward batch."""

import ast
from pathlib import Path

import pytest
import torch


class _Model:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_new_tokens, temperature, top_k):
        self.calls.append((prompt.shape, max_new_tokens, temperature, top_k))
        return torch.zeros((prompt.size(0), prompt.size(1) + max_new_tokens), dtype=torch.long)


def _sampler():
    file = Path(__file__).resolve().parents[1] / "sample.py"
    module = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    fn = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "generate_samples")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(file), "exec"), namespace)
    return namespace["generate_samples"]


@pytest.mark.parametrize("num_samples,sample_batch_size,expected", [
    (7, 3, [3, 3, 1]),
    (2, 8, [2]),
    (4, 1, [1, 1, 1, 1]),
    (0, 4, []),
])
def test_bounded_sample_batches_preserve_output_count(num_samples, sample_batch_size, expected):
    sample = _sampler()
    model = _Model()
    prompt = torch.tensor([[1, 2, 3]])
    outputs = list(sample(model, prompt, num_samples, sample_batch_size, 5, 0.8, 20))
    assert len(outputs) == num_samples
    assert [shape[0] for shape, _, _, _ in model.calls] == expected
    assert all(t.size(0) == 8 for t in outputs)
    assert all(info[1:] == (5, 0.8, 20) for info in model.calls)


def test_sampling_rejects_nonpositive_batch_size():
    sample = _sampler()
    with pytest.raises(ValueError, match="positive"):
        list(sample(_Model(), torch.ones(1, 4, dtype=torch.long), 2, 0, 3, 1.0, None))
