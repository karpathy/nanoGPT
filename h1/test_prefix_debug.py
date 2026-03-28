import torch, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
from model import GPT, GPTConfig, SoftPrefix

device = 'cuda'

# load from existing checkpoint — avoids transformers import
ckpt = torch.load('/kaggle/working/h1_L100_m1/ckpt.pt', map_location=device)
gptconf = GPTConfig(**ckpt['model_args'])
model = GPT(gptconf)
state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
model.load_state_dict(state)
model.eval().to(device)
print("Model loaded from checkpoint")

# real WikiText-2 tokens
data = np.memmap('data/wikitext2/val.bin', dtype=np.uint16, mode='r')
X = torch.from_numpy(data[0:512].astype('int64')).unsqueeze(0).to(device)
Y = torch.from_numpy(data[1:513].astype('int64')).unsqueeze(0).to(device)

# test 1: no prefix
with torch.no_grad():
    _, loss_no_prefix = model(X, Y)
print(f"no prefix:     {loss_no_prefix.item():.4f}  ← should be ~3.84")

# test 2: cached prefix
prefix = SoftPrefix(100, gptconf.n_layer, gptconf.n_head, gptconf.n_embd, device)
with torch.no_grad():
    prefix.build_cache(model)
    _, loss_cached = model(X, Y, prefix_kvs=prefix.cached_kv)
print(f"cached prefix: {loss_cached.item():.4f}  ← should be close to 3.84")

# test 3: fresh KV (update step path)
prefix_kvs = []
x_p = prefix.P.unsqueeze(0)
with torch.no_grad():
    for block in model.transformer.h:
        B, T, C = x_p.shape
        qkv = block.attn.c_attn(x_p)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, prefix.n_head, prefix.head_dim).transpose(1, 2)
        v = v.view(B, T, prefix.n_head, prefix.head_dim).transpose(1, 2)
        prefix_kvs.append((k, v))
        x_p = x_p + block.attn(block.ln_1(x_p))
        x_p = x_p + block.mlp(block.ln_2(x_p))
    _, loss_fresh = model(X, Y, prefix_kvs=prefix_kvs)
print(f"fresh prefix:  {loss_fresh.item():.4f}  ← should be close to 3.84")

# test 4: mask sanity
T, L = 4, 3
prefix_mask = torch.ones(T, L, dtype=torch.bool, device=device)
causal_mask = torch.ones(T, T, dtype=torch.bool, device=device).tril()
full_mask    = torch.cat([prefix_mask, causal_mask], dim=1)
attn_mask    = torch.zeros(T, L+T, device=device, dtype=torch.float32)
attn_mask    = attn_mask.masked_fill(~full_mask, float('-inf'))
print(f"\nmask row 0: {attn_mask[0].tolist()}")
print(f"mask row 1: {attn_mask[1].tolist()}")
print(f"mask row 2: {attn_mask[2].tolist()}")
print(f"mask row 3: {attn_mask[3].tolist()}")
print("expected: all prefix cols = 0.0, sequence cols = lower triangular")