# Day 4: Standard Reproducibility — GPT-2 Fine-tuning on WikiText-2

## 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | GPT-2 (124M) |
| 数据集 | wikitext-2-raw-v1 |
| Epochs | 1 |
| Learning rate | 5e-5 |
| Warmup steps | 100 |
| Batch size | 4 |
| GPU | NVIDIA GeForce RTX 5090 (32 GB) |

---

## 实验结果

| Run | Seed | Train Loss | Eval Loss | Perplexity | 训练时长 | wandb URL |
|-----|------|-----------|-----------|------------|---------|-----------|
| gpt2-wt2-seed42   | 42   | 3.2642 | 3.0712 | 21.57 | 1:04 | https://wandb.ai/grace2056320586-guangdong-technionisrael-institute-of-te/nanogpt/runs/pelqc04i |
| gpt2-wt2-seed123  | 123  | 3.2654 | 3.0726 | 21.60 | 1:05 | https://wandb.ai/grace2056320586-guangdong-technionisrael-institute-of-te/nanogpt/runs/hejr4279 |
| gpt2-wt2-seed2024 | 2024 | 3.2656 | 3.0717 | 21.58 | 1:05 | https://wandb.ai/grace2056320586-guangdong-technionisrael-institute-of-te/nanogpt/runs/x2s6qoz2 |
| **均值** | — | **3.2651** | **3.0718** | **21.58** | — | — |
| **标准差** | — | **0.0006** | **0.0006** | **0.013** | — | — |

---

## 与官方数字对比

| 来源 | PPL |
|------|-----|
| GPT-2 论文（zero-shot，wikitext-2） | ~29.4 |
| 本次实验（fine-tuned，1 epoch） | **21.58** |
| 差值 | -7.8 |

### 分析

1. **fine-tune vs zero-shot**：本次实验对 GPT-2 在 wikitext-2 上做了 1 epoch 的监督微调，而论文报告的 29.4 是**零样本（zero-shot）**结果，即直接用预训练模型不做任何微调。因此 PPL 从 29.4 降到 21.58 是合理的——模型见过了训练集的文本，预测能力自然提升。

2. **复现性**：3 个不同 seed（42、123、2024）的 PPL 分别为 21.57 / 21.60 / 21.58，标准差仅 0.013，说明训练结果**高度稳定**，随机因素（权重初始化、数据 shuffle）对最终指标影响极小。

3. **结果是否合理**：目标区间为 PPL 20~30，本次结果 21.58 完全符合预期，验证了实验流程正确。

---

## 异常记录

- `--overwrite_output_dir` 参数在新版 transformers 中已移除，训练脚本中去掉该参数后正常运行。
- 服务器无法直接访问 GitHub（443 端口超时），模型/数据集通过 `HF_ENDPOINT=https://hf-mirror.com` 镜像下载，已缓存在 `/root/autodl-tmp/cache/huggingface`。
