"""Verify that context cropping releases old tensor storage without changing logits."""

import torch

from model import GPT, GPTConfig


def test_crop_block_size_copies_embeddings_out_of_large_storage():
    model = GPT(GPTConfig(block_size=64, vocab_size=128, n_layer=1, n_head=4, n_embd=32, dropout=0))
    model.eval()
    idx = torch.randint(128, (2, 8))
    with torch.no_grad():
        before, _ = model(idx, targets=idx)

    original_storage = model.transformer.wpe.weight.untyped_storage()
    old_ptr = original_storage.data_ptr()
    model.crop_block_size(8)

    assert model.transformer.wpe.weight.shape == (8, 32)
    assert model.transformer.wpe.weight.untyped_storage().data_ptr() != old_ptr
    assert model.transformer.wpe.weight.untyped_storage().nbytes() == 8 * 32 * 4

    with torch.no_grad():
        after, _ = model(idx, targets=idx)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_crop_block_size_releases_causal_mask_storage():
    model = GPT(GPTConfig(block_size=64, vocab_size=128, n_layer=1, n_head=4, n_embd=32, dropout=0))
    attn = model.transformer.h[0].attn
    # Force the mask-bearing fallback path for storage accounting, independent
    # of whether the local PyTorch build supports scaled-dot-product attention.
    mask = torch.tril(torch.ones(64, 64)).view(1, 1, 64, 64)
    attn.register_buffer("bias", mask)
    old_ptr = attn.bias.untyped_storage().data_ptr()

    model.crop_block_size(8)

    assert attn.bias.shape == (1, 1, 8, 8)
    assert attn.bias.untyped_storage().nbytes() == 8 * 8 * 4
    assert attn.bias.untyped_storage().data_ptr() != old_ptr
    torch.testing.assert_close(attn.bias[0, 0], torch.tril(torch.ones(8, 8)))
