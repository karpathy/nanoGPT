import torch
from model import GPT, SoftPrefix

device = 'cuda'
model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
for p in model.parameters():
    p.requires_grad = False
model.eval()
model.to(device)

# test 1: no prefix — should give ~3.84 on random tokens
X = torch.randint(0, 50257, (2, 64), device=device)
Y = torch.randint(0, 50257, (2, 64), device=device)
with torch.no_grad():
    _, loss_no_prefix = model(X, Y)
print(f'loss without prefix: {loss_no_prefix.item():.4f}')

# test 2: with prefix — should give similar or lower loss
prefix = SoftPrefix(100, model.config.n_layer, model.config.n_head, model.config.n_embd, device)
with torch.no_grad():
    prefix.build_cache(model)
with torch.no_grad():
    _, loss_with_prefix = model(X, Y, prefix_kvs=prefix.cached_kv)
print(f'loss with prefix:    {loss_with_prefix.item():.4f}')
print(f'difference:          {loss_with_prefix.item() - loss_no_prefix.item():.4f}')