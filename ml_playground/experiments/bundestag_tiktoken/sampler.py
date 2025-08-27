from __future__ import annotations

from ml_playground.config import SamplerConfig
from ml_playground.experiments.protocol import (
    Sampler as _SamplerProto,
    SampleReport,
)
from ml_playground.sampler import sample as _core_sample


class BundestagTiktokenSampler(_SamplerProto):
    def sample(self, cfg: SamplerConfig) -> SampleReport:  # type: ignore[override]
        _core_sample(cfg)
        return SampleReport(
            created_files=(),
            updated_files=(),
            skipped_files=(),
            messages=("[bundestag_tiktoken] sampling finished",),
        )
