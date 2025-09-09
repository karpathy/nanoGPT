"""ml_playground.sampler: sampling utilities.

Device seeding/TF32 is centrally handled in the CLI. This module constructs
device, dtype, and autocast contexts locally without exposing legacy shims.
"""

from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol
import logging
import torch
from torch import autocast

from ml_playground.checkpoint import Checkpoint, CheckpointManager
from ml_playground.config import (
    ModelConfig,
    SamplerConfig,
    READ_POLICY_BEST,
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


class Sampler(Protocol):
    def __call__(self, cfg: SamplerConfig) -> None: ...


def _load_checkpoint(
    out_dir: Path,
    device: str,
    logger: logging.Logger,
    read_policy: str,
) -> Checkpoint:
    """Load model checkpoint.

    Strict: surface errors to caller.
    """
    ckpt_mgr = CheckpointManager(out_dir=out_dir)
    if read_policy == READ_POLICY_BEST:
        return ckpt_mgr.load_best_checkpoint(device=device, logger=logger)
    # Strict: default/latest only
    return ckpt_mgr.load_latest_checkpoint(device=device, logger=logger)


def sample(cfg: SamplerConfig) -> None:
    """Sample from a trained model."""
    # --- Setup -------------------------------------------------------------------
    runtime_cfg = cfg.runtime
    sample_cfg = cfg.sample
    if runtime_cfg is None:
        raise ValueError("Runtime configuration is missing")

    setup_logging(str(runtime_cfg.out_dir))
    logger = logging.getLogger(__name__)

    # --- Set random seeds -------------------------------------------------------
    torch.manual_seed(runtime_cfg.seed)
    torch.cuda.manual_seed(runtime_cfg.seed)

    # --- Device setup -----------------------------------------------------------
    device_type = "cuda" if "cuda" in runtime_cfg.device else "cpu"
    pt_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[runtime_cfg.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else autocast(device_type=device_type, dtype=pt_dtype)
    )

    # --- Load checkpoint --------------------------------------------------------
    checkpoint = _load_checkpoint(
        runtime_cfg.out_dir,
        runtime_cfg.device,
        logger,
        read_policy=runtime_cfg.checkpointing.read_policy,
    )

    # --- Model setup ------------------------------------------------------------
    model_cfg = ModelConfig(**checkpoint.model_args)
    model = GPT(model_cfg)
    model.load_state_dict(checkpoint.model, strict=False)
    model.eval()
    model.to(runtime_cfg.device)
    if runtime_cfg.compile:
        model = torch.compile(model)  # type: ignore

    # --- Tokenizer setup --------------------------------------------------------
    tokenizer = setup_tokenizer(runtime_cfg.out_dir)
    if not tokenizer:
        raise DataError(
            f"Tokenizer metadata not found in {runtime_cfg.out_dir} (expected meta.pkl)."
        )

    # --- Sampling ---------------------------------------------------------------
    start_text = sample_cfg.start
    if isinstance(start_text, str) and start_text.startswith("FILE:"):
        prompt_path = Path(start_text[5:])
        try:
            start_text = prompt_path.read_text(encoding="utf-8")
        except Exception as e:  # pragma: no cover - robust IO guard
            logger.error(f"Failed to read prompt file {prompt_path}: {e}")
            return
    start_ids = tokenizer.encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=runtime_cfg.device)[None, ...]

    logger.info("Sampling...")
    with torch.no_grad():
        with ctx:
            for k in range(sample_cfg.num_samples):
                y = model.generate(  # type: ignore
                    x,
                    sample_cfg.max_new_tokens,
                    temperature=sample_cfg.temperature,
                    top_k=sample_cfg.top_k,
                )
                output = tokenizer.decode(y[0].tolist())
                logger.info(output)
                logger.info("---------------")


__all__ = [
    "Sampler",
    "sample",
]
