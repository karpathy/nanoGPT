from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ml_playground.configuration.models import PreparerConfig, SharedConfig
from ml_playground.configuration.models import DataConfig
from ml_playground.data_pipeline.transforms.tokenization import prepare_with_tokenizer
from ml_playground.data_pipeline.transforms.io import (
    diff_file_states,
    snapshot_file_states,
    write_bin_and_meta,
)
from ml_playground.data_pipeline.transforms.tokenization import (
    TokenizerKind,
    coerce_tokenizer_type,
)
from ml_playground.core.error_handling import DataError
from ml_playground.tokenizer import create_tokenizer
from ml_playground.tokenizer_protocol import Tokenizer


@dataclass(frozen=True)
class PreparationOutcome:
    created_files: tuple[Path, ...]
    updated_files: tuple[Path, ...]
    skipped_files: tuple[Path, ...]
    metadata: dict[str, Any]


class _PreparationPipeline:
    def __init__(self, cfg: PreparerConfig, shared: SharedConfig) -> None:
        self._cfg = cfg
        self._shared = shared
        self._logger = cfg.logger

    @property
    def cfg(self) -> PreparerConfig:
        return self._cfg

    @property
    def shared(self) -> SharedConfig:
        return self._shared

    def run(self) -> PreparationOutcome:
        tokenizer_kind: TokenizerKind = self._resolve_tokenizer_type()
        tokenizer = create_tokenizer(tokenizer_kind)
        raw_text = self._load_raw_text()
        return self.prepare_from_text(raw_text, tokenizer)

    def prepare_from_text(
        self,
        text: str,
        tokenizer: Tokenizer,
        *,
        split: float | None = None,
        meta_extras: dict[str, Any] | None = None,
    ) -> PreparationOutcome:
        data_cfg = self._resolve_data_config()
        outputs = self._output_paths(data_cfg)
        before = snapshot_file_states(outputs)

        ratio = float(split) if split is not None else self._default_split()
        train_arr, val_arr, meta, tokenizer = prepare_with_tokenizer(
            text,
            tokenizer,
            split=ratio,
        )

        if meta_extras:
            meta.update(meta_extras)

        write_bin_and_meta(
            self._shared.dataset_dir,
            train_arr,
            val_arr,
            meta,
            logger=self._logger,
            data_cfg=data_cfg,
        )

        created, updated, skipped = diff_file_states(outputs, before)
        return PreparationOutcome(
            created_files=tuple(created),
            updated_files=tuple(updated),
            skipped_files=tuple(skipped),
            metadata=meta,
        )

    def _resolve_tokenizer_type(self) -> TokenizerKind:
        return coerce_tokenizer_type(self._cfg.tokenizer_type)

    def _resolve_data_config(self) -> DataConfig | None:
        data_cfg = self._cfg.extras.get("data_config")
        if data_cfg is None:
            return None

        if isinstance(data_cfg, DataConfig):
            return data_cfg
        raise DataError(
            "prepare.extras.data_config must be a DataConfig instance when provided"
        )

    def _default_split(self) -> float:
        raw = self._cfg.extras.get("split")
        if raw is None:
            return 0.9
        try:
            ratio = float(raw)
        except (TypeError, ValueError) as exc:
            raise DataError(f"Invalid split ratio in extras: {raw!r}") from exc
        if not (0.0 <= ratio <= 1.0):
            raise DataError(f"split ratio must be within [0.0, 1.0]; received {ratio}")
        return ratio

    def _load_raw_text(self) -> str:
        raw_text_path = self._cfg.raw_text_path
        if raw_text_path is not None:
            return Path(raw_text_path).read_text(encoding="utf-8")
        raise DataError("No raw text path provided in preparer config")

    def _output_paths(self, data_cfg: DataConfig | None) -> list[Path]:
        if data_cfg is not None:
            return [
                data_cfg.train_path(self._shared.dataset_dir),
                data_cfg.val_path(self._shared.dataset_dir),
                data_cfg.meta_path(self._shared.dataset_dir),
            ]
        return [
            self._shared.dataset_dir / "train.bin",
            self._shared.dataset_dir / "val.bin",
            self._shared.dataset_dir / "meta.pkl",
        ]

    def output_snapshot(self, paths: Iterable[Path]) -> dict:
        return snapshot_file_states(paths)


def create_pipeline(cfg: PreparerConfig, shared: SharedConfig) -> _PreparationPipeline:
    return _PreparationPipeline(cfg, shared)


__all__ = [
    "PreparationOutcome",
    "create_pipeline",
    "snapshot_file_states",
    "diff_file_states",
]
