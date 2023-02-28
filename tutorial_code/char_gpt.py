import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F


global data, train_data, valid_data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Checking all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
vocab_set = "".join(chars)

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Train and test splits
train_size = 0.9
data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_size * len(data))
train_data = data[:n]
valid_data = data[n:]


# Encoder: take a string, output a list of integers
def encode(s):
    return [stoi[c] for c in s]

# Decoder: take a list of integers, output a string
def decode(l):
    return ''.join([itos[i] for i in l])


def get_batch(split: str, block_size: int = 8, batch_size: int = 4, device: str = None):
    """ Gets a randomized batch from the split of data chosen.

    Arguments
    ---------
    split : str, {"train", "valid"}
    block_size : int
        The context length for predictions, that is, a sentence length
    batch_size : int
        The batch size, that is, the number of sentences
    """
    # generate a small batch of data of inputs x and targets y
    assert split in ["train", "valid"]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = train_data if split == 'train' else valid_data
    # generating random indices as markers in the full text document
    # such that they are a starting point to the sentence of length
    # `block_size` that will be a data point in the batch
    ix = torch.randint(
        low=0, high=len(data) - block_size, size=(batch_size,)
    )
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix`
    x = torch.stack([data[i:i+block_size] for i in ix])
    # extracting a sentence of length `block_size` for every
    # random starting point in `ix` + 1 (shifted to right)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters: int):
    """ Function to evaluate the model on train & valid splits.
    """
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_and_evaluate_model(
    model: nn.Module,
    block_size: int,
    batch_size: int,
    optimizer: torch.optim = None,
    num_train_steps: int = 10000,
    verbosity_len: int = 1000,
    eval_iters: int = 500,
    plot_loss: str = True,
    device: str = "cpu",
    **kwargs
):
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=kwargs["learning_rate"]
        )

    train_losses = [np.inf]
    valid_losses = [np.inf]

    for iter in tqdm(range(num_train_steps)):

        # sample a batch of data
        xb, yb = get_batch('train', block_size, batch_size, device)

        # evaluate loss on the batch
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)

        # gradient update
        loss.backward()
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets
        if iter % verbosity_len == 0 or iter == num_train_steps - 1:
            _losses = estimate_loss(model, eval_iters)
            train_losses.append(_losses['train'])
            valid_losses.append(_losses['valid'])
            print()
            print(
                f"step {iter}: train loss {_losses['train']:.4f}, "\
                f"val loss {_losses['valid']:.4f}"
            )

    if plot_loss:
        plot_losses(train_losses, verbosity_len, valid_losses)
    return losses


def generate_from_model(
    model: nn.Module, batch_num: int, sentence_len: int, start_str: str = None
):
    # sampling a start token and generating a batch of it as context
    if start_str is None:
        start_token = np.random.randint(VOCAB_SIZE)
        print(f"Start token: {decode([start_token])}")
        context = torch.zeros((batch_num, 1), dtype=torch.long, device=DEVICE)
        # setting the first token of the batch to the sampled start token
        context[:, 0] = start_token
    else:
        start_token = encode(start_str)
        print(f"Start token: {decode(start_token)}")
        # generating batch of sentences with the start token
        context = torch.tensor(start_token, dtype=torch.long, device=DEVICE)
        context = context.repeat(batch_num, 1)
    # will generate the next sentence_len characters for each of the start token
    out = model.generate(
        context, max_new_tokens=sentence_len, block_size=BLOCK_SIZE
    )
    print(out.shape)
    return out


def decode_and_print_batch(batch):
    for b in range(batch.shape[0]):
        print(f"\nBatch ID: {b}")
        print(decode(batch[b].tolist()))
    print()


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size: int, n_embed: int, head_size: int):
        """
        Arguments
        ---------
        block_size: int
            The sentence length allowed
        n_embed: int
            The size (dimension) of the token embeddings
        head_size: int
            The size (dimension) of the output of an attention head
        """
        super().__init__()

        self.block_size = block_size  # equivalent to T
        self.n_embed = n_embed
        self.head_size = head_size  # equivalent to C

        self.key = nn.Linear(self.n_embed, self.head_size, bias=False)
        self.query = nn.Linear(self.n_embed, self.head_size, bias=False)
        self.value = nn.Linear(self.n_embed, self.head_size, bias=False)

        self.register_buffer(
            'tril', torch.tril(torch.ones(self.block_size, self.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape  # B: batch size; T: block size; C: embedding size
        k = self.key(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)
        q = self.query(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        # performing `scaled` attention
        wei *= head_size ** -(1 / 2)  # scaling by `1/sqrt(head size)`

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(
        self, block_size: int, n_embed: int, head_size: int, num_heads: int
    ):
        """
        Arguments
        ---------
        block_size: int
            The sentence length allowed
        n_embed: int
            The size (dimension) of the token embeddings
        head_size: int
            The size (dimension) of the output of an attention head
        num_heads: int
            The number of single attention heads that together form
            one multi-headed attention layer
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(block_size, n_embed, head_size) for _ in range(num_heads)]
        )
        # linear FC layer
        self.proj = nn.Linear(head_size * num_heads, n_embed)

    def forward(self, x):
        # simply stack multiple heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # B: batch size; T: block size; C: embedding size; H: head_size * num_heads
        out = self.proj(out)  # (B, T, H) @ (H, C) -> (B, T, C)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(
            self,
            n_embed: int,
            wide_factor: int = 4,
            activation: str = "relu",
            dropout: float = 0.0
        ):
        super().__init__()
        self.activation = nn.ReLU if activation == "relu" else nn.GELU
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(n_embed, wide_factor * n_embed),
            self.activation(),
            nn.Linear(wide_factor * n_embed, n_embed),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(
        self,
        block_size: int,
        n_embed: int,
        num_heads: int,
        wide_factor: int = 4,
        activation: str = "relu",  # could also be "gelu"
        dropout: float = 0.0,
        prenormalize: bool = False
    ):
        super().__init__()
        # setting head_size to be a factor of other dimensions
        head_size = n_embed // num_heads
        # the multi-headed self-attention (msa)
        self.msa = MultiHeadAttention(block_size, n_embed, head_size, num_heads)
        self.ffwd = FeedForward(n_embed, wide_factor, activation, dropout)

        self.prenormalize = prenormalize
        if prenormalize:
            self.pre_ln = nn.LayerNorm(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        if self.prenormalize:
            # normalizes inputs before passing it through the attention block
            x = x + self.msa( self.pre_ln(x) )
        else:
            x = x + self.msa(x)
        # norm after attention
        x = self.ln1(x)
        # feed-forward
        x = x + self.ffwd(x)
        # norm after feed-forward
        x = self.ln2(x)
        return x


class CharGPT(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            n_layers: int,
            block_size: int,
            n_embed: int,
            num_heads: int,
            wide_factor: int = 4,
            activation: str = "relu",  # could also be "gelu"
            dropout: float = 0.0,
            prenormalize: bool = False,
            device: str = None
    ):
        super().__init__()
        # each token directly reads off the logits for the next
        # token from a lookup table
        # Note attention does not have any notion of colocation
        # of characters/words and this is important for lms
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(
                block_size=block_size,
                n_embed=n_embed,
                num_heads=num_heads,
                wide_factor=wide_factor,
                activation=activation,
                dropout=dropout,
                prenormalize=prenormalize,
            ) for _ in range(n_layers)]  # stacks the layers of Transformer blocks
        )
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm (has bias)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        self.device = device
        if self.device is None:
            self.device = "gpu" if torch.cuda.is_available() else "cpu"

    def forward(self, idx, targets=None):
        # B: batch_size, T: block_size, C: embedding_size
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        # fixing positional inputs and learning an embedding over it
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        # adding the positional embeddings across the token embedding batch
        x = tok_emb + pos_emb  # (B,T,C)
        # forward pass through the Transformer layers
        x = self.blocks(x)  # (B,T,C)
        # final layernorm
        x = self.ln_f(x)  # (B,T,C)
        # projecting to the vocabulary
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size, verbose: bool = False):
        # B: batch_size, T: block_size, C:
        # idx is (B, T) array of indices in the current context

        model.eval()

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        model.train()
        return idx

