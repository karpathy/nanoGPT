import numpy as np
import argparse
import pickle
import torch

class LayerNorm:
    def __init__(self, ndim, bias=False):
        self.weight = np.ones((ndim,))
        self.bias = np.zeros((ndim,)) if bias else None

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        x = (x - mean) / (std + 1e-5)
        if self.bias is not None:
            x = x * self.weight + self.bias
        return x

    def load_weights(self, weights):
        self.weight = weights['weight']
        if self.bias is not None:
            self.bias = weights['bias']

class CausalSelfAttention:
    def __init__(self, n_embd, n_head, block_size):
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

        self.c_attn_q = np.random.randn(n_embd, n_embd)
        self.c_attn_k = np.random.randn(n_embd, n_embd)
        self.c_attn_v = np.random.randn(n_embd, n_embd)
        self.c_proj = np.random.randn(n_embd, n_embd)

        self.bias = np.tril(np.ones((block_size, block_size))).reshape(1, 1, block_size, block_size)

    def forward(self, x, cache=None):
        B, T, C = x.shape

        q = np.dot(x, self.c_attn_q)
        k = np.dot(x, self.c_attn_k)
        v = np.dot(x, self.c_attn_v)

        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

        if cache is not None:
            k = np.concatenate([cache['k'], k], axis=2)
            v = np.concatenate([cache['v'], v], axis=2)
        else:
            cache = {'k': k, 'v': v}

        att = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(k.shape[-1])
        att = np.where(self.bias[:,:,:T,:T] == 0, -np.inf, att)
        att = np.exp(att - np.max(att, axis=-1, keepdims=True))
        att = att / np.sum(att, axis=-1, keepdims=True)

        y = np.matmul(att, v)
        y = y.transpose(1, 2).reshape(B, T, C)
        y = np.dot(y, self.c_proj)

        return y, cache

    def load_weights(self, weights):
        self.c_attn_q = weights['c_attn_q']
        self.c_attn_k = weights['c_attn_k']
        self.c_attn_v = weights['c_attn_v']
        self.c_proj = weights['c_proj']

class MLP:
    def __init__(self, n_embd):
        self.c_fc = np.random.randn(n_embd, 4 * n_embd)
        self.c_proj = np.random.randn(4 * n_embd, n_embd)

    def forward(self, x):
        x = np.dot(x, self.c_fc)
        x = np.maximum(0, x)  # GELU activation approximation
        x = np.dot(x, self.c_proj)
        return x

    def load_weights(self, weights):
        self.c_fc = weights['c_fc']
        self.c_proj = weights['c_proj']

class Block:
    def __init__(self, n_embd, n_head, block_size):
        self.ln_1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x, cache=None):
        attn_out, new_cache = self.attn.forward(self.ln_1.forward(x), cache)
        x = x + attn_out
        x = x + self.mlp.forward(self.ln_2.forward(x))
        return x, new_cache

    def load_weights(self, weights):
        self.ln_1.load_weights(weights['ln_1'])
        self.attn.load_weights(weights['attn'])
        self.ln_2.load_weights(weights['ln_2'])
        self.mlp.load_weights(weights['mlp'])

class Transformer:
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size):
        self.wte = np.random.randn(vocab_size, n_embd)
        self.wpe = np.random.randn(block_size, n_embd)

        self.blocks = [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)

        self.lm_head = self.wte  # Weight tying

    def forward(self, idx, cache=None):
        B, T = idx.shape
        x = self.wte[idx] + self.wpe[:T]

        new_cache = []
        for i, block in enumerate(self.blocks):
            x, block_cache = block.forward(x, cache[i] if cache is not None else None)
            new_cache.append(block_cache)

        x = self.ln_f.forward(x)
        return x, new_cache

    def load_weights(self, weights):
        # Load the embedding weights
        self.wte = weights['transformer.wte.weight']
        self.wpe = weights['transformer.wpe.weight']

        # Load the weights for each block
        for i, block in enumerate(self.blocks):
            block_weights = {
                'ln_1': {
                    'weight': weights[f'transformer.h.{i}.ln_1.weight']
                },
                'attn': {
                    'c_attn_q': weights[f'transformer.h.{i}.attn.c_attn_q.weight'],
                    'c_attn_k': weights[f'transformer.h.{i}.attn.c_attn_k.weight'],
                    'c_attn_v': weights[f'transformer.h.{i}.attn.c_attn_v.weight'],
                    'c_proj': weights[f'transformer.h.{i}.attn.c_proj.weight']
                },
                'ln_2': {
                    'weight': weights[f'transformer.h.{i}.ln_2.weight']
                },
                'mlp': {
                    'c_fc': weights[f'transformer.h.{i}.mlp.c_fc.weight'],
                    'c_proj': weights[f'transformer.h.{i}.mlp.c_proj.weight']
                }
            }
            block.load_weights(block_weights)

        # Load the final layer normalization weights
        self.ln_f.load_weights({'weight': weights['transformer.ln_f.gain']})

        # Tie the lm_head weights to the wte
        self.lm_head = self.wte

def load_model_weights(file_path):
    with open(file_path, 'rb') as f:
        weights = pickle.load(f)

    # Print all the keys in the weights file
    print("Keys in the loaded weights file:")
    for key in weights.keys():
        print(key)

    return weights



def summarization_stage(model, idx):
    logits, cache = model.forward(idx)
    next_token = np.argmax(logits[:, -1, :], axis=-1)
    return next_token, cache

def generation_stage(model, idx, max_new_tokens, cache, block_size):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.shape[1] <= block_size else idx[:, -block_size:]
        logits, cache = model.forward(idx_cond, cache)
        next_token = np.argmax(logits[:, -1, :], axis=-1)
        idx = np.concatenate((idx, next_token[:, None]), axis=1)
    return idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--vocab_size', type=int, default=50304)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--weights_path', type=str, required=True)
    args = parser.parse_args()

    model = Transformer(args.vocab_size, args.n_layer, args.n_head, args.n_embd, args.block_size)

    # Load weights
    weights = load_model_weights(args.weights_path)
    model.load_weights(weights)

    # Sample input: just a sequence of zeros
    idx = np.zeros((1, 1), dtype=int)

    # Summarization stage
    next_token, cache = summarization_stage(model, idx)
    idx = np.concatenate((idx, next_token[:, None]), axis=1)

    # Generation stage
    generated_sequence = generation_stage(model, idx, args.max_new_tokens, cache, args.block_size)
    print("Generated sequence:", generated_sequence)

if __name__ == "__main__":
    main()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--vocab_size', type=int, default=50304)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--weights_path', type=str, required=True)
    args = parser.parse_args()

    model = Transformer(args.vocab_size, args.n_layer, args.n_head, args.n_embd, args.block_size)

    # Load weights and print the keys
    weights = load_model_weights(args.weights_path)
    model.load_weights(weights)

    # Sample input: just a sequence of zeros
    idx = np.zeros((1, 1), dtype=int)

    # Summarization stage
    next_token, cache = summarization_stage(model, idx)
    idx = np.concatenate((idx, next_token[:, None]), axis=1)

    # Generation stage
    generated_sequence = generation_stage(model, idx, args.max_new_tokens, cache, args.block_size)
    print("Generated sequence:", generated_sequence)

if __name__ == "__main__":
    main()

