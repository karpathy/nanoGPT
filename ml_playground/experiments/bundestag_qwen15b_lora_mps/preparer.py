from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ml_playground.prepare import PreparerConfig
from ml_playground.experiments.protocol import (
    Preparer as _PreparerProto,
    PrepareReport,
)


class BundestagQwen15bLoraMpsPreparer(_PreparerProto):
    """Minimal preparer to satisfy CLI integration for this preset.

    This preset rides on a generic HF+PEFT integration for training/sampling.
    For preparation, we currently only ensure the configured dataset directory
    exists so downstream steps can locate it. A richer pipeline can be added
    later to actually build tokenizer/artifacts from raw data.
    """

    def prepare(self, cfg: PreparerConfig) -> PrepareReport:  # type: ignore[override]
        # Determine dataset directory: use local folder under this preset
        exp_dir = Path(__file__).resolve().parent
        ds_dir = (exp_dir / "datasets").resolve()

        # Track side-effects (creation/updates) for user feedback
        tracked: list[Path] = [ds_dir]
        before = _snapshot(tracked)

        # Ensure dataset directory exists
        ds_dir.mkdir(parents=True, exist_ok=True)

        created, updated, skipped = _diff(tracked, before)
        msgs = (
            f"[bundestag_qwen15b_lora_mps] ensured dataset directory at {ds_dir}",
            f"[bundestag_qwen15b_lora_mps.outputs.created] {[str(p) for p in created]}",
            f"[bundestag_qwen15b_lora_mps.outputs.updated] {[str(p) for p in updated]}",
            f"[bundestag_qwen15b_lora_mps.outputs.skipped] {[str(p) for p in skipped]}",
        )
        return PrepareReport(
            created_files=tuple(created),
            updated_files=tuple(updated),
            skipped_files=tuple(skipped),
            messages=msgs,
        )


def _snapshot(paths: Iterable[Path]) -> dict[Path, tuple[bool, float, int]]:
    m: dict[Path, tuple[bool, float, int]] = {}
    for p in paths:
        try:
            if p.exists():
                st = p.stat()
                m[p] = (True, st.st_mtime, st.st_size)
            else:
                m[p] = (False, 0.0, 0)
        except Exception:
            m[p] = (False, 0.0, 0)
    return m


def _diff(paths: Iterable[Path], before: dict[Path, tuple[bool, float, int]]):
    created: list[Path] = []
    updated: list[Path] = []
    skipped: list[Path] = []
    for p in paths:
        existed, mtime, size = before.get(p, (False, 0.0, 0))
        try:
            if p.exists():
                st = p.stat()
                if not existed:
                    created.append(p)
                elif st.st_mtime != mtime or st.st_size != size:
                    updated.append(p)
                else:
                    skipped.append(p)
        except Exception:
            if p.exists() and not existed:
                created.append(p)
    return created, updated, skipped
