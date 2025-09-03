from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ml_playground.error_handling import CheckpointError


def _atomic_save(obj: Any, path: Path, atomic: bool) -> None:
    if atomic:
        # Atomic save via rename
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp_path)
        tmp_path.rename(path)
    else:
        torch.save(obj, path)


@dataclass  # type: ignore
class Checkpoint:
    """A strongly-typed checkpoint object."""

    model: Dict[str, Any]
    optimizer: Dict[str, Any]
    model_args: Dict[str, Any]
    iter_num: int
    best_val_loss: float
    config: Dict[str, Any]
    ema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        result = {
            "model": self.model,
            "optimizer": self.optimizer,
            "model_args": self.model_args,
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if self.ema is not None:
            result["ema"] = self.ema
        return result

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to attributes."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator to check for attribute existence."""
        return hasattr(self, key)


@dataclass
class _CkptInfo:
    """Internal checkpoint metadata for management."""

    path: Path
    metric: float
    iter_num: int
    created_at: float


class CheckpointManager:
    """A utility class for managing checkpoints with advanced features."""

    def __init__(
        self, out_dir: Path, atomic: bool = True, keep_last: int = 1, keep_best: int = 1
    ):
        self.out_dir = out_dir
        self.atomic = atomic
        if keep_last < 0 or keep_best < 0:
            raise CheckpointError(
                f"Invalid checkpoint keep policy: keep_last={keep_last}, keep_best={keep_best} (must be >= 0)"
            )
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.last_checkpoints: List[_CkptInfo] = []
        self.best_checkpoints: List[_CkptInfo] = []
        # Discover any existing checkpoints so behavior persists across restarts
        self._discover_existing()

    def _discover_existing(self) -> None:
        """Scan filesystem for existing rotated checkpoints and rebuild state."""
        # last: ckpt_last_XXXXXXXX.pt
        for p in sorted(self.out_dir.glob("ckpt_last_*.pt")):
            # iter from filename suffix
            stem = p.stem  # e.g., ckpt_last_00000010
            parts = stem.split("_")
            if len(parts) < 3:
                raise CheckpointError(f"Malformed last-checkpoint filename: {p.name}")
            iter_str = parts[-1]
            try:
                it = int(iter_str)
            except Exception as e:
                raise CheckpointError(
                    f"Could not parse iteration from last-checkpoint filename {p.name}: {e}"
                ) from e
            try:
                created = p.stat().st_mtime
            except Exception as e:
                raise CheckpointError(f"Failed to stat checkpoint file {p}: {e}") from e
            self.last_checkpoints.append(_CkptInfo(p, float("inf"), it, created))
        # best: ckpt_best_XXXXXXXX_*.pt (metric may be encoded)
        for p in sorted(self.out_dir.glob("ckpt_best_*.pt")):
            stem = p.stem  # e.g., ckpt_best_00000010_1.234567
            parts = stem.split("_")
            if len(parts) < 3:
                raise CheckpointError(f"Malformed best-checkpoint filename: {p.name}")
            try:
                it = int(parts[2])
            except Exception as e:
                raise CheckpointError(
                    f"Could not parse iteration from best-checkpoint filename {p.name}: {e}"
                ) from e
            metric = float("inf")
            if len(parts) >= 4:
                try:
                    metric = float(parts[3])
                except Exception as e:
                    raise CheckpointError(
                        f"Could not parse metric from best-checkpoint filename {p.name}: {e}"
                    ) from e
            try:
                created = p.stat().st_mtime
            except Exception as e:
                raise CheckpointError(f"Failed to stat checkpoint file {p}: {e}") from e
            self.best_checkpoints.append(_CkptInfo(p, metric, it, created))

    def _update_stable_pointer(self, rotated_path: Path, stable_filename: str) -> None:
        """Point the stable filename to the rotated file via symlink if possible, else copy."""
        stable_path = self.out_dir / stable_filename
        if stable_path.exists() or stable_path.is_symlink():
            try:
                stable_path.unlink()
            except Exception as e:
                raise CheckpointError(
                    f"Failed to remove existing stable checkpoint pointer {stable_path}: {e}"
                ) from e
        # Try to create a relative symlink; on failure, try hard copy; if both fail, raise
        try:
            stable_path.symlink_to(rotated_path.name)
            return
        except Exception as symlink_err:
            try:
                shutil.copy2(rotated_path, stable_path)
                return
            except Exception as copy_err:
                raise CheckpointError(
                    f"Failed to update stable checkpoint pointer {stable_path}: symlink error={symlink_err}; copy error={copy_err}"
                ) from copy_err

    def save_checkpoint(
        self,
        checkpoint: Checkpoint,
        base_filename: str,
        metric: float,
        iter_num: int,
        logger: Optional[logging.Logger] = None,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint with metadata and manage last and best checkpoints.

        Returns the rotated checkpoint path that was written.
        """
        # Determine rotated filename based on kind
        if is_best:
            rotated_name = f"ckpt_best_{iter_num:08d}_{metric:.6f}.pt"
        else:
            rotated_name = f"ckpt_last_{iter_num:08d}.pt"
        path = self.out_dir / rotated_name

        # Save the checkpoint
        _atomic_save(checkpoint.to_dict(), path, self.atomic)

        ckpt_info = _CkptInfo(path, metric, iter_num, time.time())

        # Manage last checkpoints
        if self.keep_last > 0 and not is_best:
            # Remove any existing checkpoint with the same path
            self.last_checkpoints = [
                ckpt for ckpt in self.last_checkpoints if ckpt.path != path
            ]
            self.last_checkpoints.append(ckpt_info)
            # Keep only the specified number of last checkpoints
            if len(self.last_checkpoints) > self.keep_last:
                # Remove oldest checkpoints
                to_remove = self.last_checkpoints[
                    : len(self.last_checkpoints) - self.keep_last
                ]
                self.last_checkpoints = self.last_checkpoints[
                    len(self.last_checkpoints) - self.keep_last :
                ]

                # Delete the files
                for ckpt in to_remove:
                    try:
                        ckpt.path.unlink()
                        # Also remove sidecar file if it exists
                        sidecar = ckpt.path.with_suffix(ckpt.path.suffix + ".json")
                        if sidecar.exists():
                            sidecar.unlink()
                        if logger:
                            logger.info(f"Removed old last checkpoint: {ckpt.path}")
                    except Exception as e:
                        raise CheckpointError(
                            f"Failed to remove old last checkpoint {ckpt.path}: {e}"
                        ) from e

        # Manage best checkpoints
        if is_best and self.keep_best > 0:
            # Remove any existing checkpoint with the same path
            self.best_checkpoints = [
                ckpt for ckpt in self.best_checkpoints if ckpt.path != path
            ]
            self.best_checkpoints.append(ckpt_info)
            # Sort by metric (assuming lower is better by default)
            self.best_checkpoints.sort(key=lambda x: x.metric, reverse=False)

            # Keep only the specified number of best checkpoints
            if len(self.best_checkpoints) > self.keep_best:
                # Remove worst checkpoints
                to_remove = self.best_checkpoints[self.keep_best :]
                self.best_checkpoints = self.best_checkpoints[: self.keep_best]

                # Delete the files
                for ckpt in to_remove:
                    try:
                        ckpt.path.unlink()
                        # Also remove sidecar file if it exists
                        sidecar = ckpt.path.with_suffix(ckpt.path.suffix + ".json")
                        if sidecar.exists():
                            sidecar.unlink()
                        if logger:
                            logger.info(f"Removed old best checkpoint: {ckpt.path}")
                    except Exception as e:
                        raise CheckpointError(
                            f"Failed to remove old best checkpoint {ckpt.path}: {e}"
                        ) from e

        # Strict mode: do NOT create or update any stable checkpoint pointers
        # such as ckpt_last.pt or ckpt_best.pt. Only rotated checkpoints are produced.

        if logger:
            logger.info(f"Saved checkpoint to {path}")
        return path

    def load_latest_checkpoint(
        self, device: str, logger: Optional[logging.Logger] = None
    ) -> Checkpoint:
        """Load the latest checkpoint from the last checkpoints list."""
        if not self.last_checkpoints:
            # try discovering from disk
            self._discover_existing()
            if not self.last_checkpoints:
                raise CheckpointError(
                    f"No last checkpoints discovered in {self.out_dir}"
                )

        # Get the most recent checkpoint
        latest_ckpt = max(self.last_checkpoints, key=lambda x: x.created_at)

        try:
            checkpoint_dict = torch.load(str(latest_ckpt.path), map_location=device)
        except Exception as e:
            if logger:
                logger.error(f"Error loading checkpoint from {latest_ckpt.path}: {e}")
            raise CheckpointError(
                f"Failed to load checkpoint from {latest_ckpt.path}: {e}"
            ) from e

        if not isinstance(checkpoint_dict, dict):
            raise CheckpointError("Checkpoint file does not contain a dictionary")

        # Validate required keys
        required_keys = [
            "model",
            "optimizer",
            "model_args",
            "iter_num",
            "best_val_loss",
            "config",
        ]
        for key in required_keys:
            if key not in checkpoint_dict:
                raise CheckpointError(f"Checkpoint missing required key: {key}")

        # Create Checkpoint object
        checkpoint = Checkpoint(
            model=checkpoint_dict["model"],
            optimizer=checkpoint_dict["optimizer"],
            model_args=checkpoint_dict["model_args"],
            iter_num=checkpoint_dict["iter_num"],
            best_val_loss=checkpoint_dict["best_val_loss"],
            config=checkpoint_dict["config"],
            ema=checkpoint_dict.get("ema"),
        )

        if logger:
            logger.info(f"Loaded checkpoint from {latest_ckpt.path}")
        return checkpoint

    def load_best_checkpoint(
        self, device: str, logger: Optional[logging.Logger] = None
    ) -> Checkpoint:
        """Load the best checkpoint from the best checkpoints list."""
        if not self.best_checkpoints:
            self._discover_existing()
            if not self.best_checkpoints:
                raise CheckpointError(
                    f"No best checkpoints discovered in {self.out_dir}"
                )

        # Get the best checkpoint (lowest metric)
        best_ckpt = min(self.best_checkpoints, key=lambda x: x.metric)

        try:
            checkpoint_dict = torch.load(str(best_ckpt.path), map_location=device)
        except Exception as e:
            if logger:
                logger.error(
                    f"Error loading best checkpoint from {best_ckpt.path}: {e}"
                )
            raise CheckpointError(
                f"Failed to load best checkpoint from {best_ckpt.path}: {e}"
            ) from e

        if not isinstance(checkpoint_dict, dict):
            raise CheckpointError("Checkpoint file does not contain a dictionary")

        # Validate required keys
        required_keys = [
            "model",
            "optimizer",
            "model_args",
            "iter_num",
            "best_val_loss",
            "config",
        ]
        for key in required_keys:
            if key not in checkpoint_dict:
                raise CheckpointError(f"Checkpoint missing required key: {key}")

        # Create Checkpoint object
        checkpoint = Checkpoint(
            model=checkpoint_dict["model"],
            optimizer=checkpoint_dict["optimizer"],
            model_args=checkpoint_dict["model_args"],
            iter_num=checkpoint_dict["iter_num"],
            best_val_loss=checkpoint_dict["best_val_loss"],
            config=checkpoint_dict["config"],
            ema=checkpoint_dict.get("ema"),
        )

        if logger:
            logger.info(f"Loaded checkpoint from {best_ckpt.path}")
        return checkpoint
