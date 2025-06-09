import os
import torch
import torch.nn as nn
from tqdm import tqdm
import re

# Load and concatenate all .txt files
data_dir = 'output'
all_text = ''
for fname in os.listdir(data_dir):
    if fname.endswith('.txt'):
        with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
            all_text += f.read() + '\n'

# Preprocess data: Normalize whitespace and remove special characters/numbers
all_text = ' '.join(all_text.split())  # Normalize whitespace
all_text = re.sub(r'[^a-zA-Z\s]', '', all_text)  # Keep only letters and spaces

# Tokenize and build vocabulary (character-level)
chars = sorted(list(set(all_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(all_text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Batch preparation
block_size = 256  # Increased context length
batch_size = 64   # Increased batch size

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

# Improved TinyGPT model
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.embed_dropout = nn.Dropout(0.2)  # Increased dropout
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dropout=0.2) 
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.embed(idx)
        tok_emb = self.embed_dropout(tok_emb)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_embed(pos)[None, :, :]
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

# Training setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TinyGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)  # Lowered learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
loss_fn = nn.CrossEntropyLoss()

# Training loop
max_iters = 10000  # Increased iterations
eval_interval = 100

for iter in tqdm(range(max_iters)):
    model.train()
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)
    loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch('val')
            xb, yb = xb.to(device), yb.to(device)
            val_logits = model(xb)
            val_loss = loss_fn(val_logits.view(-1, vocab_size), yb.view(-1))
        print(f"Iter {iter}, train loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}")

# Improved text generation
def generate(model, start_text, max_new_tokens=200, temperature=0.5):
    model.eval()
    idx = torch.tensor(encode(start_text), dtype=torch.long, device=device)[None, :]
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # Lower temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return decode(idx[0].tolist())

# Test generation with your example’s starting line
print("\nGenerated sample:")
print(generate(model, "OPERATOR Monday  at  PM", max_new_tokens=200, temperature=0.5))