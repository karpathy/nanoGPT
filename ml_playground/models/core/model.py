from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
from torch.nn import functional as F

from ml_playground.configuration.models import ModelConfig
from ml_playground.models.core.config import GPTConfig, build_gpt_config
from ml_playground.models.core.inference import estimate_model_mfu, generate_tokens
from ml_playground.models.core.optimization import (
    configure_optimizers as _configure_optimizers,
)
from ml_playground.models.layers.block import Block
from ml_playground.models.layers.normalization import LayerNorm
from ml_playground.models.utils.init import init_transformer_weights


class GPT(nn.Module):
    """GPT model backed by the modular `ml_playground.models` hierarchy."""

    def __init__(self, config: ModelConfig | GPTConfig, logger) -> None:
        super().__init__()
        if isinstance(config, ModelConfig):
            config = build_gpt_config(config)
        config = cast(GPTConfig, config)

        vocab_size = config.vocab_size
        if vocab_size is None:
            raise ValueError("GPTConfig.vocab_size must be set")

        self.config: GPTConfig = config
        self.logger = logger

        self.token_embeddings = nn.Embedding(vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        self.token_embeddings.weight.requires_grad_(True)

        self.apply(init_transformer_weights)
        for pn, param in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

        if self.logger:
            self.logger.info(
                "number of parameters: %.2fM",
                self.get_num_params() / 1e6,
            )

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(param.numel() for param in self.parameters())
        if non_embedding:
            n_params -= self.token_embeddings.weight.numel()
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                "Cannot forward sequence of length %s, block size is only %s"
                % (seq_len, self.config.block_size)
            )

        positions = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)

        tok_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(positions)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        if block_size > self.config.block_size:
            raise ValueError("block_size cannot be increased dynamically")
        self.config.block_size = block_size
        self.position_embeddings = nn.Embedding(block_size, self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - legacy API parity
        raise NotImplementedError("from_pretrained is not supported in this port")

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        return _configure_optimizers(
            self,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
            device_type=device_type,
            logger=self.logger,
        )

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        return estimate_model_mfu(self, fwdbwd_per_iter, dt)

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        return generate_tokens(
            self,
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )


__all__ = ["GPT"]
