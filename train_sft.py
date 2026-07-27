"""
SFT 训练脚本：在 Stanford Alpaca 上对 GPT-2 124M 做指令微调。

核心区别于 train.py：
  - loss 只在 response token 上计算（prompt token 被 mask 掉），
    这是 SFT 的标准做法，避免模型把算力花在“学习复述指令”上。
  - 从 GPT-2 124M 预训练权重初始化。
  - 单 GPU 训练（124M 足够小）；如需多卡可参考 train.py 的 DDP 包装。

用法：
  python data/alpaca/prepare.py     # 先准备数据
  python train_sft.py               # 开始 SFT
  python sample.py --out_dir=out-alpaca --start="Below is an instruction..."

命令行可覆盖任意超参，例如：
  python train_sft.py --batch_size=8 --learning_rate=5e-5 --max_iters=2000
"""
import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 默认超参（GPT-2 124M + Alpaca SFT）
out_dir = 'out-alpaca'
eval_interval = 50
log_interval = 1
eval_iters = 20
always_save_checkpoint = False
init_from = 'gpt2'  # 124M

dataset = 'alpaca'
batch_size = 4
gradient_accumulation_steps = 8       # 等效 batch=32
block_size = 512                      # Alpaca 样本短，512 够用；裁剪 wpe 省显存
max_iters = 1000                      # 约 1 个 epoch（见下方 token 估算）

# adamw
learning_rate = 3e-5                  # SFT 用小学习率，防灾难性遗忘
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0

# lr 调度：warmup + cosine decay
decay_lr = True
warmup_iters = 50
lr_decay_iters = 1000
min_lr = 3e-6

# model
dropout = 0.1                         # SFT 小数据建议开 dropout 防过拟合
bias = False

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
# -----------------------------------------------------------------------------

# 允许命令行 / config 文件覆盖
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# ----- 环境初始化 -----
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# ----- 数据加载（带 mask）-----
data_dir = os.path.join('data', dataset)


def get_batch(split):
    """随机采样 block_size 长度的窗口，返回 x, y, mask。
    约定与 train.py 一致：x = tokens[i:i+B], y = tokens[i+1:i+1+B]，
    因此 mask 也按 y 对齐：mask = masks[i+1:i+1+B]。
    mask=1 表示该位置是 response token，参与 loss。
    """
    if split == 'train':
        tokens = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        masks = np.memmap(os.path.join(data_dir, 'train_labels.bin'), dtype=np.uint8, mode='r')
    else:
        tokens = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        masks = np.memmap(os.path.join(data_dir, 'val_labels.bin'), dtype=np.uint8, mode='r')

    ix = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((tokens[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((tokens[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    m = torch.stack([torch.from_numpy((masks[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        m = m.pin_memory().to(device, non_blocking=True)
    else:
        x, y, m = x.to(device), y.to(device), m.to(device)
    return x, y, m


# ----- 模型初始化（从 GPT-2 124M）-----
print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
model_args = dict(n_layer=None, n_head=None, n_embd=None, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
model = GPT.from_pretrained(init_from, dict(dropout=dropout))
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = getattr(model.config, k)

# 裁剪 block_size（GPT-2 原生 1024，这里裁到 512 省 wpe 显存）
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)


# ----- masked SFT loss -----
def sft_loss(logits, targets, mask):
    """logits: (B, T, V)  targets: (B, T)  mask: (B, T) ∈ {0,1}
    logits[t] 预测第 t+1 个 token，与 targets[t] 对齐（nanoGPT 约定，无需再 shift）。
    只对 mask=1 的位置计算 cross-entropy，再按 response token 数归一化。
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction='none')
    loss = loss.view(B, T) * mask.float()
    n = mask.float().sum().clamp(min=1.0)
    return loss.sum() / n


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, M = get_batch(split)
            with ctx:
                # 传 targets=Y 才能拿到全序列 logits（见 model.py forward）
                logits, _ = model(X, Y)
                loss = sft_loss(logits, Y, M)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ----- 训练循环 -----
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration: {tokens_per_iter:,} (response 占比约 30%)")

best_val_loss = 1e9
t0 = time.time()
X, Y, M = get_batch('train')

for iter_num in range(max_iters + 1):
    # 设置学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # 评估 + 存档
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                raw_model = model.module if hasattr(model, 'module') else model
                ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))

    # 前向 + 反向（梯度累积）
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, _ = model(X, Y)
            loss = sft_loss(logits, Y, M) / gradient_accumulation_steps
        # 异步预取下一个 batch
        X, Y, M = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # 日志
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")

print("SFT done.")
