from __future__ import annotations

from pathlib import Path
from ml_playground.config import SamplerConfig
from ml_playground.experiments.protocol import (
    Sampler as _SamplerProto,
    SampleReport,
)


class BundestagQwen15bLoraMpsSampler(_SamplerProto):
    def sample(self, cfg: SamplerConfig) -> SampleReport:  # type: ignore[override]
        """Minimal no-op sampler placeholder for the preset.

        This repository does not embed the full HF/PEFT sampling loop for Qwen.
        We provide a stub that ensures the configured out_dir exists so that the
        CLI flow can run end-to-end and tests observing side-effects can pass.
        """
        out_dir = getattr(getattr(cfg, "runtime", object()), "out_dir", None)
        if isinstance(out_dir, Path):
            out_dir.mkdir(parents=False, exist_ok=True)
        return SampleReport(
            created_files=(),
            updated_files=(),
            skipped_files=(),
            messages=(
                "[bundestag_qwen15b_lora_mps] no-op sampler; external HF/PEFT pipeline expected outside repository",
            ),
        )
