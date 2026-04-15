import math

import pytest
import torch

from model import GPT, GPTConfig


def tiny_config(**overrides):
    base = dict(
        block_size=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )
    base.update(overrides)
    return GPTConfig(**base)


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return GPT(tiny_config())


def test_gptconfig_defaults_instantiate():
    cfg = GPTConfig()
    assert cfg.block_size == 1024
    assert cfg.vocab_size == 50304
    assert cfg.n_layer == 12


def test_forward_shapes_with_targets(tiny_model):
    B, T = 4, 8
    idx = torch.randint(0, 64, (B, T))
    targets = torch.randint(0, 64, (B, T))
    logits, loss = tiny_model(idx, targets)
    assert logits.shape == (B, T, 64)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_forward_inference_only_returns_last_position(tiny_model):
    idx = torch.randint(0, 64, (2, 8))
    logits, loss = tiny_model(idx)
    assert logits.shape == (2, 1, 64)
    assert loss is None


def test_forward_is_deterministic_under_fixed_seed():
    torch.manual_seed(123)
    m1 = GPT(tiny_config())
    torch.manual_seed(123)
    m2 = GPT(tiny_config())

    idx = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(7))
    m1.eval()
    m2.eval()
    with torch.no_grad():
        l1, _ = m1(idx)
        l2, _ = m2(idx)
    assert torch.equal(l1, l2)


def test_forward_rejects_too_long_sequence(tiny_model):
    idx = torch.randint(0, 64, (1, 17))
    with pytest.raises(AssertionError):
        tiny_model(idx)


def test_generate_token_count_and_shape(tiny_model):
    tiny_model.eval()
    idx = torch.randint(0, 64, (1, 4))
    out = tiny_model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=5)
    assert out.shape == (1, 14)
    assert (out[:, :4] == idx).all()


def test_generate_respects_block_size_cropping(tiny_model):
    """Starting context exceeds block_size; generate must crop and not blow up."""
    tiny_model.eval()
    idx = torch.randint(0, 64, (1, 20))  # > block_size=16
    out = tiny_model.generate(idx, max_new_tokens=5, top_k=5)
    assert out.shape == (1, 25)


def test_crop_block_size_shrinks_wpe(tiny_model):
    assert tiny_model.transformer.wpe.weight.shape == (16, 32)
    tiny_model.crop_block_size(8)
    assert tiny_model.config.block_size == 8
    assert tiny_model.transformer.wpe.weight.shape == (8, 32)
    idx = torch.randint(0, 64, (1, 8))
    logits, _ = tiny_model(idx)
    assert logits.shape == (1, 1, 64)


def test_configure_optimizers_partitions_by_ndim(tiny_model):
    opt = tiny_model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-3, betas=(0.9, 0.95), device_type="cpu"
    )
    assert len(opt.param_groups) == 2
    decay_group, nodecay_group = opt.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert nodecay_group["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() < 2 for p in nodecay_group["params"])


def test_estimate_mfu_returns_finite_float(tiny_model):
    mfu = tiny_model.estimate_mfu(fwdbwd_per_iter=1, dt=0.01)
    assert isinstance(mfu, float)
    assert math.isfinite(mfu)
    assert mfu > 0


def test_get_num_params_excludes_position_embeddings_by_default(tiny_model):
    n_with = tiny_model.get_num_params(non_embedding=False)
    n_without = tiny_model.get_num_params(non_embedding=True)
    pos_emb_count = tiny_model.transformer.wpe.weight.numel()
    assert n_with - n_without == pos_emb_count


def test_loss_decreases_on_overfit_batch():
    torch.manual_seed(0)
    model = GPT(tiny_config())
    model.train()
    idx = torch.randint(0, 64, (4, 16))
    targets = torch.randint(0, 64, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(20):
        _, loss = model(idx, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())

    assert losses[-1] < losses[0] * 0.5, f"loss did not decrease enough: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_weight_tying_between_wte_and_lm_head(tiny_model):
    assert tiny_model.transformer.wte.weight.data_ptr() == tiny_model.lm_head.weight.data_ptr()
