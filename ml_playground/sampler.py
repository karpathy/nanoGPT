"""ml_playground.sampler: sampling utilities.

Device seeding/TF32 is centrally handled in the CLI. This module constructs
device, dtype, and autocast contexts locally without exposing legacy shims.
"""

from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
from typing import cast
import logging
import torch
from torch import autocast

from ml_playground.checkpoint import Checkpoint, CheckpointManager
from ml_playground.config import (
    ModelConfig,
    SamplerConfig,
    READ_POLICY_BEST,
    SharedConfig,
)
from ml_playground.error_handling import DataError, setup_logging
from ml_playground.model import GPT
from ml_playground.prepare import setup_tokenizer


"""
Centralized sampling utilities for ml_playground experiments.

This module provides standardized utilities for model sampling including:
- Checkpoint loading with proper error handling
- Error handling with centralized exception types

All experiments should use these utilities to ensure consistency and proper error handling.
"""


class Sampler:
    def __init__(self, cfg: SamplerConfig, shared: SharedConfig):
        """Initialize the sampler."""
        self.cfg = cfg
        self.shared = shared
        self.runtime_cfg = cfg.runtime
        self.sample_cfg = cfg.sample

        if self.runtime_cfg is None:
            raise ValueError("Runtime configuration is missing")

        self.out_dir = shared.sample_out_dir
        setup_logging(str(self.out_dir))
        self.logger = logging.getLogger(__name__)

        torch.manual_seed(self.runtime_cfg.seed)
        torch.cuda.manual_seed(self.runtime_cfg.seed)

        self.device_type = "cuda" if "cuda" in self.runtime_cfg.device else "cpu"
        pt_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.runtime_cfg.dtype]
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else autocast(device_type=self.device_type, dtype=pt_dtype)
        )

        checkpoint = self._load_checkpoint()
        self.model = self._setup_model(checkpoint)
        self.tokenizer = self._setup_tokenizer()

    def _load_checkpoint(self) -> Checkpoint:
        """Load model checkpoint."""
        ckpt_mgr = CheckpointManager(out_dir=self.out_dir)
        if self.runtime_cfg.checkpointing.read_policy == READ_POLICY_BEST:
            return ckpt_mgr.load_best_checkpoint(
                device=self.runtime_cfg.device, logger=self.logger
            )
        return ckpt_mgr.load_latest_checkpoint(
            device=self.runtime_cfg.device, logger=self.logger
        )

    def _setup_model(self, checkpoint: Checkpoint) -> GPT:
        model_cfg = ModelConfig(**checkpoint.model_args)
        model = GPT(model_cfg)
        model.load_state_dict(checkpoint.model, strict=False)
        model.eval()
        model.to(self.runtime_cfg.device)
        if self.runtime_cfg.compile:
            # torch.compile returns Any; cast to GPT for static typing
            model = cast(GPT, torch.compile(model))
        return model

    def _setup_tokenizer(self):
        tokenizer = setup_tokenizer(self.out_dir)
        if not tokenizer:
            raise DataError(
                f"Tokenizer metadata not found in {self.out_dir} (expected meta.pkl)."
            )
        return tokenizer

    def run(self) -> None:
        """Run the sampling process."""
        start_text = self.sample_cfg.start
        if isinstance(start_text, str) and start_text.startswith("FILE:"):
            prompt_path = Path(start_text[5:])
            try:
                start_text = prompt_path.read_text(encoding="utf-8")
            except Exception as e:
                self.logger.error(f"Failed to read prompt file {prompt_path}: {e}")
                return
        start_ids = self.tokenizer.encode(start_text)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.runtime_cfg.device)[
            None, ...
        ]

        self.logger.info("Sampling...")
        with torch.no_grad():
            with self.ctx:
                for k in range(self.sample_cfg.num_samples):
                    y = self.model.generate(
                        x,
                        self.sample_cfg.max_new_tokens,
                        temperature=self.sample_cfg.temperature,
                        top_k=self.sample_cfg.top_k,
                    )
                    output = self.tokenizer.decode(y[0].tolist())
                    self.logger.info(output)
                    self.logger.info("---------------")


def sample(cfg: SamplerConfig, shared: SharedConfig | None = None) -> None:
    """Sample from a trained model."""
    if shared is None:
        if cfg.runtime is None:
            raise ValueError("Runtime configuration is missing")
        out_dir = cfg.runtime.out_dir
        shared = SharedConfig(
            experiment="unknown",
            config_path=out_dir / "config.toml",
            project_home=out_dir.parent.parent,
            dataset_dir=out_dir,  # Assume same as out_dir for backward compat
            train_out_dir=out_dir,
            sample_out_dir=out_dir,
        )
    sampler = Sampler(cfg, shared)
    sampler.run()
    """Sample from a trained model."""


__all__ = [
    "Sampler",
    "sample",
]
