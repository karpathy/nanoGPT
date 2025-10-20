from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import torch
from torch.serialization import pickle as _torch_pickle  # type: ignore[attr-defined]
from ml_playground.core.error_handling import CheckpointError, CheckpointLoadError
from ml_playground.core.logging_protocol import LoggerLike

TorchUnpicklingError = _torch_pickle.UnpicklingError  # type: ignore[attr-defined]


__all__ = ["Checkpoint", "CheckpointManager", "CheckpointDependencies"]


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


@dataclass
class CheckpointDependencies:
    torch_load: Callable[..., Any]
    add_safe_globals: Callable[[Iterable[Any]], None] | None
    path_stat: Callable[[Path], os.stat_result]
    path_unlink: Callable[[Path], None]
    posix_path_cls: type | None
    unlink_supports_missing_ok: bool

    @classmethod
    def default(cls) -> "CheckpointDependencies":
        def _path_stat(path: Path) -> os.stat_result:
            return path.stat()

        def _path_unlink(path: Path) -> None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        posix_cls: type | None = None
        try:
            import pathlib

            posix_cls = getattr(getattr(pathlib, "_local", None), "PosixPath", None)
        except Exception:
            posix_cls = None

        supports_missing_ok = True
        probe_path = Path(".checkpoint_unlink_probe")
        try:
            probe_path.touch(exist_ok=True)
            probe_path.unlink(missing_ok=True)
        except TypeError:
            supports_missing_ok = False
        except OSError:
            supports_missing_ok = False
        finally:
            if probe_path.exists():
                try:
                    probe_path.unlink()
                except OSError:
                    pass

        return cls(
            torch_load=torch.load,
            add_safe_globals=add_safe_globals,
            path_stat=_path_stat,
            path_unlink=_path_unlink,
            posix_path_cls=posix_cls,
            unlink_supports_missing_ok=supports_missing_ok,
        )


def _expect_mapping(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError(
            f"Checkpoint field '{field}' must be a mapping",
            reason=f"Observed type {type(value).__name__}",
            rationale="Checkpoint serialization stores dictionaries for composite payload sections",
        )
    return dict(value)


def _expect_type(value: Any, field: str, expected_type: ExpectedTypes) -> Any:
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected_names = ", ".join(t.__name__ for t in expected_type)
        else:
            expected_names = expected_type.__name__
        raise CheckpointError(
            f"Checkpoint field '{field}' expected {expected_names}, got {type(value).__name__}",
            reason="Type mismatch in checkpoint payload",
            rationale="Checkpoint validation enforces stable types so reloading preserves training state",
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
                f"Checkpoint payload missing required fields: {', '.join(sorted(missing))}",
                reason=f"Absent keys: {', '.join(sorted(missing))}",
                rationale="Restoring training requires all critical checkpoint sections to be present",
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
        self,
        out_dir: Path,
        atomic: bool = True,
        keep_last: int = 1,
        keep_best: int = 1,
        *,
        deps: CheckpointDependencies | None = None,
    ):
        self.out_dir = out_dir
        self.atomic = atomic
        if keep_last < 0 or keep_best < 0:
            raise CheckpointError(
                f"Invalid checkpoint keep policy: keep_last={keep_last}, keep_best={keep_best} (must be >= 0)",
                reason="Retention counts cannot be negative",
                rationale="Checkpoint rotation policies rely on non-negative keep windows to manage files deterministically",
            )
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.last_checkpoints: List[_CkptInfo] = []
        self.best_checkpoints: List[_CkptInfo] = []
        self._deps = deps or CheckpointDependencies.default()
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
                    f"Could not parse iteration from last-checkpoint filename {p.name}: {e}",
                    reason="Filename suffix does not encode an integer iteration",
                    rationale="Checkpoint discovery depends on canonical naming to rebuild manager state",
                ) from e
            created = self._stat_checkpoint_file(p)
            self.last_checkpoints.append(_CkptInfo(p, float("inf"), it, created))
        for p in sorted(self.out_dir.glob("ckpt_best_*.pt")):
            stem = p.stem  # e.g., ckpt_best_00000010_1.234567
            parts = stem.split("_")
            if len(parts) < 3:
                raise CheckpointError(
                    f"Unexpected best-checkpoint filename format: {p.name}",
                    reason="Filename lacks iteration/metric segments",
                    rationale="Best-checkpoint retention requires canonical ckpt_best_<iter>_<metric>.pt names",
                )
            try:
                it = int(parts[2])
            except ValueError as e:
                raise CheckpointError(
                    f"Could not parse iteration from best-checkpoint filename {p.name}: {e}",
                    reason="Iteration segment is not an integer",
                    rationale="Consistent numbering lets the manager sort best checkpoints by creation",
                ) from e
            metric = float("inf")
            if len(parts) >= 4:
                try:
                    metric = float(parts[3])
                except ValueError as e:
                    raise CheckpointError(
                        f"Could not parse metric from best-checkpoint filename {p.name}: {e}",
                        reason="Metric segment is not a float",
                        rationale="Best checkpoint ordering depends on parsing the recorded metric",
                    ) from e
            created = self._stat_checkpoint_file(p)
            self.best_checkpoints.append(_CkptInfo(p, metric, it, created))

    def _stat_checkpoint_file(self, path: Path) -> float:
        """Return the modification time for ``path`` with uniform error handling."""
        try:
            return self._deps.path_stat(path).st_mtime
        except OSError as e:
            raise CheckpointError(
                f"Failed to stat checkpoint file {path}: {e}",
                reason=f"{e.__class__.__name__} while retrieving filesystem metadata",
                rationale="Manager must inspect filesystem timestamps to order checkpoints",
            ) from e

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
        if self.keep_last > -1 and not is_best:
            # Track the new last checkpoint and prune oldest beyond the retention window
            self.last_checkpoints = [
                ckpt for ckpt in self.last_checkpoints if ckpt.path != path
            ]
            self.last_checkpoints.append(ckpt_info)
            # Sort by creation time (oldest first)
            self.last_checkpoints.sort(key=lambda x: x.created_at)
            # If we exceed the retention policy, delete oldest from disk and list
            while len(self.last_checkpoints) > self.keep_last:
                old = self.last_checkpoints.pop(0)
                try:
                    if self._deps.unlink_supports_missing_ok:
                        old.path.unlink(missing_ok=False)
                    else:
                        self._deps.path_unlink(old.path)
                    logger.info(f"Removed old last checkpoint: {old.path}")
                except OSError as e:
                    raise CheckpointError(
                        f"Failed to remove old last checkpoint {old.path}: {e}",
                        reason=f"{e.__class__.__name__} while deleting checkpoint file",
                        rationale="Retention pruning must delete expired checkpoints to honour keep_last policy",
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
                        if self._deps.unlink_supports_missing_ok:
                            ckpt.path.unlink(missing_ok=False)
                        else:
                            self._deps.path_unlink(ckpt.path)
                        # Also remove sidecar file if it exists
                        sidecar = ckpt.path.with_suffix(ckpt.path.suffix + ".json")
                        if sidecar.exists():
                            if self._deps.unlink_supports_missing_ok:
                                sidecar.unlink(missing_ok=False)
                            else:
                                self._deps.path_unlink(sidecar)
                        logger.info(f"Removed old best checkpoint: {ckpt.path}")
                    except OSError as e:
                        raise CheckpointError(
                            f"Failed to remove old best checkpoint {ckpt.path}: {e}",
                            reason=f"{e.__class__.__name__} while deleting checkpoint file",
                            rationale="Retention pruning must delete expired best checkpoints to honour keep_best policy",
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
                    f"No last checkpoints discovered in {self.out_dir}",
                    reason="Last checkpoint rotation list is empty",
                    rationale="Loading latest checkpoint requires at least one saved checkpoint on disk",
                )

        # Get the most recent checkpoint
        latest_ckpt = max(self.last_checkpoints, key=lambda x: x.created_at)

        try:
            # PyTorch 2.6+: default weights_only=True breaks loading objects like PosixPath;
            # use safe allowlist to enable loading our known globals and set weights_only=False for tests
            add_safe_globals = self._deps.add_safe_globals
            if callable(add_safe_globals):
                posix_path_cls = self._deps.posix_path_cls
                if posix_path_cls is not None:
                    try:
                        add_safe_globals([posix_path_cls])  # type: ignore[arg-type]
                    except (RuntimeError, TypeError):
                        # Ignore duplicates or incompatible registrations
                        pass
            checkpoint_dict = self._deps.torch_load(
                str(latest_ckpt.path), map_location=device, weights_only=False
            )
        except (OSError, RuntimeError, TorchUnpicklingError) as e:
            logger.error(f"Error loading checkpoint from {latest_ckpt.path}: {e}")
            raise CheckpointLoadError(
                f"Failed to load checkpoint from {latest_ckpt.path}: {e}",
                reason=f"Deserialization raised {e.__class__.__name__}",
                rationale="Checkpoints must be readable Torch payloads to resume training",
            ) from e

        if not isinstance(checkpoint_dict, Mapping):
            raise CheckpointError(
                "Checkpoint file does not contain a mapping payload",
                reason=f"Loaded object type: {type(checkpoint_dict).__name__}",
                rationale="Checkpoint payloads must be mapping-like to hydrate strongly typed objects",
            )

        checkpoint = Checkpoint.from_payload(cast(Mapping[str, Any], checkpoint_dict))

        logger.info(f"Loaded checkpoint from {latest_ckpt.path}")
        return checkpoint

    def load_best_checkpoint(self, device: str, logger: LoggerLike) -> Checkpoint:
        """Load the best-performing checkpoint according to recorded metric."""
        if not self.best_checkpoints:
            self._discover_existing()
            if not self.best_checkpoints:
                raise CheckpointError(
                    f"No best checkpoints discovered in {self.out_dir}",
                    reason="Best checkpoint rotation list is empty",
                    rationale="Loading best checkpoint requires at least one tracked best checkpoint",
                )

        # Get the best checkpoint (lowest metric)
        best_ckpt = min(self.best_checkpoints, key=lambda x: x.metric)

        try:
            # See comment in load_latest_checkpoint regarding safe loading
            add_safe_globals = self._deps.add_safe_globals
            if callable(add_safe_globals):
                posix_path_cls = self._deps.posix_path_cls
                if posix_path_cls is not None:
                    try:
                        add_safe_globals([posix_path_cls])  # type: ignore[arg-type]
                    except (RuntimeError, TypeError):
                        pass
            checkpoint_dict = self._deps.torch_load(
                str(best_ckpt.path), map_location=device, weights_only=False
            )
        except (OSError, RuntimeError, TorchUnpicklingError) as e:
            logger.error(f"Error loading best checkpoint from {best_ckpt.path}: {e}")
            raise CheckpointLoadError(
                f"Failed to load best checkpoint from {best_ckpt.path}: {e}",
                reason=f"Deserialization raised {e.__class__.__name__}",
                rationale="Best checkpoints must remain readable Torch payloads to be promoted",
            ) from e

        if not isinstance(checkpoint_dict, Mapping):
            raise CheckpointError(
                "Checkpoint file does not contain a mapping payload",
                reason=f"Loaded object type: {type(checkpoint_dict).__name__}",
                rationale="Checkpoint payloads must be mapping-like to hydrate strongly typed objects",
            )

        checkpoint = Checkpoint.from_payload(cast(Mapping[str, Any], checkpoint_dict))

        logger.info(f"Loaded checkpoint from {best_ckpt.path}")
        return checkpoint
