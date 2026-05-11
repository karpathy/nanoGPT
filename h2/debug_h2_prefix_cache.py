import torch

from model import GPT, GPTConfig, SoftPrefix


def assert_close(name, a, b, atol=1e-4, rtol=1e-4):
    max_diff = (a - b).abs().max().item()
    print(f"{name}: max_diff={max_diff:.8f}")
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{name} mismatch: max_diff={max_diff}")


def make_tiny_model(device):
    torch.manual_seed(1337)
    config = GPTConfig(
        block_size=16,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True,
    )
    model = GPT(config).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def test_cached_matches_uncached(device):
    model = make_tiny_model(device)
    prefix = SoftPrefix(4, model.config.n_embd, device).to(device)
    prefix.eval()

    x = torch.randint(0, model.config.vocab_size, (3, 8), device=device)
    y = torch.randint(0, model.config.vocab_size, (3, 8), device=device)

    with torch.no_grad():
        logits_uncached, loss_uncached = model(x, y, prefix=prefix)
        prefix_kv = model.build_prefix_kv(prefix, batch_size=x.size(0))
        logits_cached, loss_cached = model(x, y, prefix_kv=prefix_kv)

    assert_close("logits cached vs uncached", logits_cached, logits_uncached)
    assert_close("loss cached vs uncached", loss_cached[None], loss_uncached[None])


def test_sparse_update_changes_only_on_update(device):
    model = make_tiny_model(device)
    prefix = SoftPrefix(4, model.config.n_embd, device).to(device)
    opt = torch.optim.AdamW([prefix.P], lr=1e-2)

    x = torch.randint(0, model.config.vocab_size, (3, 8), device=device)
    y = torch.randint(0, model.config.vocab_size, (3, 8), device=device)

    before_update = prefix.P.detach().clone()
    prefix.requires_grad_(True)
    _, loss = model(x, y, prefix=prefix)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    after_update = prefix.P.detach().clone()

    if torch.allclose(before_update, after_update):
        raise AssertionError("prefix did not change on update step")
    print("update step: prefix changed")

    before_frozen = prefix.P.detach().clone()
    prefix.requires_grad_(False)
    with torch.no_grad():
        prefix_kv = model.build_prefix_kv(prefix, batch_size=x.size(0))
        _, _ = model(x, y, prefix_kv=prefix_kv)
    after_frozen = prefix.P.detach().clone()

    assert_close("frozen step prefix unchanged", before_frozen, after_frozen, atol=0.0, rtol=0.0)


def test_cache_invalidates_after_prefix_change(device):
    model = make_tiny_model(device)
    prefix = SoftPrefix(4, model.config.n_embd, device).to(device)
    x = torch.randint(0, model.config.vocab_size, (3, 8), device=device)

    with torch.no_grad():
        old_kv = model.build_prefix_kv(prefix, batch_size=x.size(0))
        logits_old, _ = model(x, prefix_kv=old_kv)
        prefix.P.add_(0.25)
        new_kv = model.build_prefix_kv(prefix, batch_size=x.size(0))
        logits_new, _ = model(x, prefix_kv=new_kv)

    max_diff = (logits_old - logits_new).abs().max().item()
    print(f"cache invalidation sensitivity: max_logit_diff={max_diff:.8f}")
    if max_diff <= 1e-5:
        raise AssertionError("changing prefix did not affect cached logits")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"debug device: {device}")
    test_cached_matches_uncached(device)
    test_sparse_update_changes_only_on_update(device)
    test_cache_invalidates_after_prefix_change(device)
    print("all H2 prefix-cache checks passed")
