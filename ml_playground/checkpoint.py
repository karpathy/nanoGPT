from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, TypedDict, Union, cast

import torch
from torch.serialization import pickle as _torch_pickle  # type: ignore[attr-defined]

from ml_playground.core.error_handling import CheckpointError, CheckpointLoadError
from ml_playground.core.logging_protocol import LoggerLike

TorchUnpicklingError = _torch_pickle.UnpicklingError  # type: ignore[attr-defined]


__all__ = ["Checkpoint", "CheckpointManager"]


def _atomic_save(obj: Any, path: Path, atomic: bool) -> None:
    """Persist an object to disk, optionally using an atomic rename step."""
    if atomic:
        # Atomic save via rename
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp_path)
        tmp_path.rename(path)
    else:
        torch.save(obj, path)


StateDict = Dict[str, Any]
OptimizerState = Dict[str, Any]
ConfigDict = Dict[str, Any]
ExtrasDict = Dict[str, Any]


class CheckpointPayload(TypedDict, total=False):
    model: StateDict
    optimizer: OptimizerState
    model_args: ConfigDict
    iter_num: int
    best_val_loss: float
    config: ConfigDict
    ema: ExtrasDict


_PayloadMapping = Mapping[str, Any]
ExpectedTypes = Union[type, Tuple[type, ...]]


def _expect_mapping(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError(f"Checkpoint field '{field}' must be a mapping")
    return dict(value)


def _expect_type(value: Any, field: str, expected_type: ExpectedTypes) -> Any:
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected_names = ", ".join(t.__name__ for t in expected_type)
        else:
            expected_names = expected_type.__name__
        raise CheckpointError(
            f"Checkpoint field '{field}' expected {expected_names}, got {type(value).__name__}"
        )
    return value


@dataclass
class Checkpoint:
    """A strongly-typed checkpoint object."""

    model: StateDict
    optimizer: OptimizerState
    model_args: ConfigDict
    iter_num: int
    best_val_loss: float
    config: ConfigDict
    ema: Optional[ExtrasDict] = None

    def to_dict(self) -> CheckpointPayload:
        """Convert checkpoint to a serialization-friendly dictionary."""
        result: CheckpointPayload = {
            "model": dict(self.model),
            "optimizer": dict(self.optimizer),
            "model_args": dict(self.model_args),
            "iter_num": self.iter_num,
            "best_val_loss": float(self.best_val_loss),
            "config": dict(self.config),
        }
        if self.ema is not None:
            result["ema"] = dict(self.ema)
        return result

    @classmethod
    def from_payload(cls, payload: _PayloadMapping) -> "Checkpoint":
        """Construct a checkpoint from a serialized payload with validation."""

        required_fields = {
            "model",
            "optimizer",
            "model_args",
            "iter_num",
            "best_val_loss",
            "config",
        }
        missing = required_fields.difference(payload.keys())
        if missing:
            raise CheckpointError(
                f"Checkpoint payload missing required fields: {', '.join(sorted(missing))}"
            )

        model = _expect_mapping(payload["model"], "model")
        optimizer = _expect_mapping(payload["optimizer"], "optimizer")
        model_args = _expect_mapping(payload["model_args"], "model_args")
        config = _expect_mapping(payload["config"], "config")

        iter_num = cast(int, _expect_type(payload["iter_num"], "iter_num", int))
        best_val_loss = cast(
            float, _expect_type(payload["best_val_loss"], "best_val_loss", (int, float))
        )

        ema_raw = payload.get("ema")
        ema: Optional[ExtrasDict]
        if ema_raw is not None:
            ema = _expect_mapping(ema_raw, "ema")
        else:
            ema = None

        return cls(
            model=model,
            optimizer=optimizer,
            model_args=model_args,
            iter_num=iter_num,
            best_val_loss=float(best_val_loss),
            config=config,
            ema=ema,
        )


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
        """Scan the filesystem for rotated checkpoints and rebuild manager state."""
        for p in sorted(self.out_dir.glob("ckpt_last_*.pt")):
            # iter from filename suffix
            stem = p.stem  # e.g., ckpt_last_00000010
            parts = stem.split("_")
            iter_str = parts[-1]
            try:
                it = int(iter_str)
            except ValueError as e:
                raise CheckpointError(
                    f"Could not parse iteration from last-checkpoint filename {p.name}: {e}"
                ) from e
            try:
                created = p.stat().st_mtime
            except OSError as e:
                raise CheckpointError(f"Failed to stat checkpoint file {p}: {e}") from e
            self.last_checkpoints.append(_CkptInfo(p, float("inf"), it, created))
        for p in sorted(self.out_dir.glob("ckpt_best_*.pt")):
            stem = p.stem  # e.g., ckpt_best_00000010_1.234567
            parts = stem.split("_")
            try:
                it = int(parts[2])
            except ValueError as e:
                raise CheckpointError(
                    f"Could not parse iteration from best-checkpoint filename {p.name}: {e}"
                ) from e
            metric = float("inf")
            if len(parts) >= 4:
                try:
                    metric = float(parts[3])
                except ValueError as e:
                    raise CheckpointError(
                        f"Could not parse metric from best-checkpoint filename {p.name}: {e}"
                    ) from e
            try:
                created = p.stat().st_mtime
            except OSError as e:
                raise CheckpointError(f"Failed to stat checkpoint file {p}: {e}") from e
            self.best_checkpoints.append(_CkptInfo(p, metric, it, created))

    def save_checkpoint(
        self,
        checkpoint: Checkpoint,
        base_filename: str,
        metric: float,
        iter_num: int,
        logger: LoggerLike,
        is_best: bool = False,
    ) -> Path:
        """Persist a checkpoint and enforce retention policies for last and best entries."""
        # API compatibility: base_filename kept but unused with rotated-only scheme
        _ = base_filename
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
            # Track the new last checkpoint and prune oldest beyond the retention window
            self.last_checkpoints.append(ckpt_info)
            # Sort by creation time (oldest first)
            self.last_checkpoints.sort(key=lambda x: x.created_at)
            # If we exceed the retention policy, delete oldest from disk and list
            while len(self.last_checkpoints) > self.keep_last:
                old = self.last_checkpoints.pop(0)
                try:
                    old.path.unlink()
                    logger.info(f"Removed old last checkpoint: {old.path}")
                except OSError as e:
                    raise CheckpointError(
                        f"Failed to remove old last checkpoint {old.path}: {e}"
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
                        logger.info(f"Removed old best checkpoint: {ckpt.path}")
                    except OSError as e:
                        raise CheckpointError(
                            f"Failed to remove old best checkpoint {ckpt.path}: {e}"
                        ) from e

        # Strict mode: do NOT create or update any stable checkpoint pointers.
        # Only rotated checkpoints are produced.

        logger.info(f"Saved checkpoint to {path}")
        return path

    def load_latest_checkpoint(self, device: str, logger: LoggerLike) -> Checkpoint:
        """Load the most recent checkpoint that tracks iteration progress."""
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
            # PyTorch 2.6+: default weights_only=True breaks loading objects like PosixPath;
            # use safe allowlist to enable loading our known globals and set weights_only=False for tests
            import pathlib

            add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
            if callable(add_safe_globals):
                posix_path_cls = getattr(
                    getattr(pathlib, "_local", None), "PosixPath", None
                )
                if posix_path_cls is not None:
                    try:
                        add_safe_globals([posix_path_cls])  # type: ignore[arg-type]
                    except (RuntimeError, TypeError):
                        # Ignore duplicates or incompatible registrations
                        pass
            checkpoint_dict = torch.load(
                str(latest_ckpt.path), map_location=device, weights_only=False
            )
        except (OSError, RuntimeError, TorchUnpicklingError) as e:
            logger.error(f"Error loading checkpoint from {latest_ckpt.path}: {e}")
            raise CheckpointLoadError(
                f"Failed to load checkpoint from {latest_ckpt.path}: {e}"
            ) from e

        if not isinstance(checkpoint_dict, Mapping):
            raise CheckpointError("Checkpoint file does not contain a mapping payload")

        checkpoint = Checkpoint.from_payload(cast(Mapping[str, Any], checkpoint_dict))

        logger.info(f"Loaded checkpoint from {latest_ckpt.path}")
        return checkpoint

    def load_best_checkpoint(self, device: str, logger: LoggerLike) -> Checkpoint:
        """Load the best-performing checkpoint according to recorded metric."""
        if not self.best_checkpoints:
            self._discover_existing()
            if not self.best_checkpoints:
                raise CheckpointError(
                    f"No best checkpoints discovered in {self.out_dir}"
                )

        # Get the best checkpoint (lowest metric)
        best_ckpt = min(self.best_checkpoints, key=lambda x: x.metric)

        try:
            # See comment in load_latest_checkpoint regarding safe loading
            import pathlib

            add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
            if callable(add_safe_globals):
                posix_path_cls = getattr(
                    getattr(pathlib, "_local", None), "PosixPath", None
                )
                if posix_path_cls is not None:
                    try:
                        add_safe_globals([posix_path_cls])  # type: ignore[arg-type]
                    except (RuntimeError, TypeError):
                        pass
            checkpoint_dict = torch.load(
                str(best_ckpt.path), map_location=device, weights_only=False
            )
        except (OSError, RuntimeError, TorchUnpicklingError) as e:
            logger.error(f"Error loading best checkpoint from {best_ckpt.path}: {e}")
            raise CheckpointLoadError(
                f"Failed to load best checkpoint from {best_ckpt.path}: {e}"
            ) from e

        if not isinstance(checkpoint_dict, Mapping):
            raise CheckpointError("Checkpoint file does not contain a mapping payload")

        checkpoint = Checkpoint.from_payload(cast(Mapping[str, Any], checkpoint_dict))

        logger.info(f"Loaded checkpoint from {best_ckpt.path}")
        return checkpoint
