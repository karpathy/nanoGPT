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
from ml_playground.error_handling import DataError, FileOperationError
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
    """Generate samples from a trained `GPT` model using a strict configuration."""

    def __init__(self, cfg: SamplerConfig, shared: SharedConfig):
        """Instantiate the sampler and eagerly load required runtime state.

        Args:
            cfg: Fully validated sampler configuration produced by the CLI.
            shared: Shared experiment metadata, including dataset and output directories.

        Raises:
            ValueError: If the runtime section of the configuration is missing.
            DataError: If tokenizer metadata cannot be located.
        """
        self.cfg = cfg
        self.shared = shared
        self.runtime_cfg = cfg.runtime
        self.sample_cfg = cfg.sample

        if self.runtime_cfg is None:
            raise ValueError("Runtime configuration is missing")

        self.out_dir = shared.sample_out_dir
        # Use a stable, module-level logger name for predictable capture in tests
        self.logger = logging.getLogger("ml_playground.sampler")

        self._setup_torch_env()

        self.model = self._load_checkpoint_and_model()
        self.tokenizer = self._setup_tokenizer()

    def _setup_torch_env(self) -> None:
        """Seed global torch RNG state and prepare autocast context."""
        torch.manual_seed(self.runtime_cfg.seed)
        # Guard CUDA-specific calls for non-CUDA environments
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.runtime_cfg.seed)
        except (RuntimeError, AssertionError, AttributeError):
            pass

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

    def _load_checkpoint_and_model(self) -> GPT:
        """Load the configured checkpoint and materialize a `GPT` model."""
        checkpoint = self._load_checkpoint()
        return self._init_model_from_checkpoint(checkpoint)

    def _load_checkpoint(self) -> Checkpoint:
        """Load a checkpoint according to the configured read policy."""
        ckpt_mgr = CheckpointManager(out_dir=self.out_dir)
        if self.runtime_cfg.checkpointing.read_policy == READ_POLICY_BEST:
            return ckpt_mgr.load_best_checkpoint(
                device=self.runtime_cfg.device, logger=self.logger
            )
        return ckpt_mgr.load_latest_checkpoint(
            device=self.runtime_cfg.device, logger=self.logger
        )

    def _init_model_from_checkpoint(self, checkpoint: Checkpoint) -> GPT:
        """Rebuild a `GPT` instance from checkpointed weights and metadata."""
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
        """Load tokenizer metadata from the sampling or dataset directories."""
        # Try the sampling out_dir first
        tokenizer = setup_tokenizer(self.out_dir)
        if tokenizer:
            return tokenizer
        # Fallback: try dataset_dir to find meta if not propagated yet
        tokenizer = setup_tokenizer(self.shared.dataset_dir)
        if tokenizer:
            return tokenizer
        raise DataError(
            f"Tokenizer metadata not found in {self.out_dir} or {self.shared.dataset_dir} (expected meta.pkl)."
        )

    def _get_start_ids(self) -> list[int]:
        """Resolve the configured prompt source and tokenize it."""
        start_text = self.sample_cfg.start
        if isinstance(start_text, str) and start_text.startswith("FILE:"):
            prompt_path = Path(start_text[5:])
            try:
                start_text = prompt_path.read_text(encoding="utf-8")
            except (OSError, IOError) as e:
                # Replace bare except with explicit file operation error
                raise FileOperationError(
                    f"Failed to read prompt file {prompt_path}: {e}"
                ) from e
        return self.tokenizer.encode(start_text)

    def run(self) -> None:
        """Generate one or more samples and stream them through the logger."""
        start_ids = self._get_start_ids()
        if not start_ids:
            return

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


__all__ = ["Sampler"]
