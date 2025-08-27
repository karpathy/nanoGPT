from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ml_playground.prepare import PreparerConfig
from ml_playground.config import TrainerConfig, SamplerConfig


@dataclass(frozen=True)
class _SideEffectBase:
    """Structured summary of observable side effects performed by a command.

    - created_files: Files that did not exist before and exist after.
    - updated_files: Files that existed before and had their mtime/size changed.
    - skipped_files: Files that existed before and were not modified.
    - messages: Human-readable notes.
    """

    created_files: tuple[Path, ...] = field(default_factory=tuple)
    updated_files: tuple[Path, ...] = field(default_factory=tuple)
    skipped_files: tuple[Path, ...] = field(default_factory=tuple)
    messages: tuple[str, ...] = field(default_factory=tuple)

    def summarize(self) -> str:
        return (
            f"created={len(self.created_files)}, "
            f"updated={len(self.updated_files)}, "
            f"skipped={len(self.skipped_files)}"
        )


@dataclass(frozen=True)
class PrepareReport(_SideEffectBase):
    pass


@dataclass(frozen=True)
class TrainReport(_SideEffectBase):
    pass


@dataclass(frozen=True)
class SampleReport(_SideEffectBase):
    pass


class Preparer(Protocol):
    def prepare(self, cfg: PreparerConfig) -> PrepareReport: ...


class Trainer(Protocol):
    def train(self, cfg: TrainerConfig) -> TrainReport: ...


class Sampler(Protocol):
    def sample(self, cfg: SamplerConfig) -> SampleReport: ...


class ExperimentIntegration(Protocol):
    def get_preparer(self) -> Preparer: ...
    def get_trainer(self) -> Trainer: ...
    def get_sampler(self) -> Sampler: ...
