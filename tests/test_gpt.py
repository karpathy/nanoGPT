from dataclasses import asdict, dataclass
import torch
import torch.nn.functional as F
from pydantic import ValidationError
from torch import nn

from gpt import *

''' Reference implemenation taken from Karpathy's nanoGPT. '''


@dataclass
class GPTConfig:
	block_size: int = 1024
	vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
	n_layer: int = 12
	n_head: int = 12
	n_embd: int = 768
	dropout: float = 0.0
	bias: bool = True

cfg = GPTConfig()


class ReferenceCausalSelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.n_embd % config.n_head == 0
		# key, query, value projections for all heads, but in a batch
		self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
		# output projection
		self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
		# regularization
		self.attn_dropout = nn.Dropout(config.dropout)
		self.resid_dropout = nn.Dropout(config.dropout)
		self.n_head = config.n_head
		self.n_embd = config.n_embd
		self.dropout = config.dropout
		# flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
		self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
		if not self.flash:
			print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
			# causal mask to ensure that attention is only applied to the left in the input sequence
			self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
										.view(1, 1, config.block_size, config.block_size))

	def forward(self, x):
		B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

		# calculate query, key, values for all heads in batch and move head forward to be the batch dim
		q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

		# causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
		if self.flash:
			# efficient attention using Flash Attention CUDA kernels
			y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
		else:
			# manual implementation of attention
			att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
			att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
			att = F.softmax(att, dim=-1)
			att = self.attn_dropout(att)
			y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
		y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

		# output projection
		y = self.resid_dropout(self.c_proj(y))
		return y


class MLP(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.c_fc	 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
		self.gelu	 = nn.GELU()
		self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		x = self.dropout(x)
		return x


class LayerNorm(nn.Module):
	""" LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

	def __init__(self, ndim, bias):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(ndim))
		self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

	def forward(self, input):
		return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
		self.attn = ReferenceCausalSelfAttention(config)
		self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
		self.mlp = MLP(config)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class RefGPT(nn.Module):

	def __init__(self, config):
		super().__init__()
		assert config.vocab_size is not None
		assert config.block_size is not None
		self.config = config

		self.transformer = nn.ModuleDict(dict(
			wte = nn.Embedding(config.vocab_size, config.n_embd),
			wpe = nn.Embedding(config.block_size, config.n_embd),
			drop = nn.Dropout(config.dropout),
			h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
			ln_f = LayerNorm(config.n_embd, bias=config.bias),
		))
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

	def forward(self, idx):
		device = idx.device
		b, t = idx.size()
		assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
		pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

		# forward the GPT model itself
		tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
		pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
		x = self.transformer.drop(tok_emb + pos_emb)
		for block in self.transformer.h:
			x = block(x)
		x = self.transformer.ln_f(x)

		logits = self.lm_head(x)

		return logits


# ========= Tests =========

torch.set_grad_enabled(False)


def test_validation_error():
	cfg_d = dict(
		max_seq_len=1024,
		vocab_size=50304,
		n_layers=12,
		n_heads=12,
		d_embd='should error',
	)

	try:
		GPTConfig(**cfg_d)
	except ValidationError as e:
	   print(e)


def test_attn():
	attn_ref = ReferenceCausalSelfAttention(cfg).to('cuda')
	attn = CausalSelfAttention(d_embd=cfg.n_embd, n_heads=cfg.n_head).to('cuda')
	attn.load_ref_weights(attn_ref)

	B, T, E = 2, 16, cfg.n_embd
	x_BTE = torch.rand([B, T, E], device='cuda')
	y_BTE = attn(x_BTE)
	y_BTE_ref = attn_ref(x_BTE)

	assert torch.allclose(y_BTE, y_BTE_ref)


def test_ffn():
	ffn_ref = MLP(cfg).to('cuda')

	d_embd = cfg.n_embd
	ffn = nn.Sequential(
		nn.Linear(d_embd, 4*d_embd),
		nn.GELU(),
		nn.Linear(4*d_embd, d_embd)
	).to('cuda')
	ffn[0].load_state_dict(ffn_ref.c_fc.state_dict())
	ffn[2].load_state_dict(ffn_ref.c_proj.state_dict())

	x_BTE = torch.rand([2, 16, cfg.n_embd], device='cuda')
	y_BTE = ffn(x_BTE)
	y_BTE_ref = ffn_ref(x_BTE)

	assert torch.allclose(y_BTE, y_BTE_ref)


def test_gpt_block():
	block_ref = Block(cfg).to('cuda')
	block = GPTBlock(d_embd=cfg.n_embd, n_heads=cfg.n_head).to('cuda')
	block.load_ref_weights(block_ref)

	x_BTE = torch.rand([2, 16, cfg.n_embd], device='cuda')
	y_BTE = block(x_BTE)
	y_BTE_ref = block_ref(x_BTE)

	assert torch.allclose(y_BTE, y_BTE_ref)


def test_gpt():
	gpt_ref = RefGPT(cfg).to('cuda')
	gpt = GPT(
		vocab_size=cfg.vocab_size,
		max_seq_len=cfg.block_size,
		d_embd=cfg.n_embd,
		n_layers=cfg.n_layer,
		n_heads=cfg.n_head
	).to('cuda')
	gpt.load_ref_weights(gpt_ref)

	idx_BT = torch.randint(cfg.vocab_size, [2, cfg.block_size], device='cuda')
	logits_BTV = gpt(idx_BT)
	logits_BTV_ref = gpt_ref(idx_BT)

	assert torch.allclose(logits_BTV, logits_BTV_ref)


def test_fp8_gpt():
	gpt_fp8 = Fp8GPT(
		vocab_size=cfg.vocab_size,
		max_seq_len=cfg.block_size,
		d_embd=cfg.n_embd,
		n_layers=cfg.n_layer,
		n_heads=cfg.n_head
	).to('cuda')
	idx_BT = torch.randint(cfg.vocab_size, [2, cfg.block_size], device='cuda')
	logits_BTV = gpt_fp8(idx_BT)
