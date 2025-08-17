from __future__ import annotations
from contextlib import AbstractContextManager, nullcontext
from typing import Tuple
import torch
from ml_playground.config import DeviceKind, DTypeKind


DTYPE_MAP: dict[DTypeKind, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class AmpContext:
    """A small wrapper that selects an appropriate autocast context.

    On CPU: use torch.cpu.amp.autocast when dtype != float32.
    On CUDA: use torch.amp.autocast(device_type="cuda", dtype=... ).
    On MPS: autocast support is limited; enable only for float16, otherwise no-op.
    """

    def __init__(self, device_type: DeviceKind, dtype: torch.dtype):
        self.device_type = device_type
        self.dtype = dtype
        self._ctx: AbstractContextManager
        if device_type == "cpu":
            enable = dtype != torch.float32
            # CPU autocast supports bfloat16 in recent torch versions, but we allow both bf16/fp16
            self._ctx = torch.amp.autocast("cpu", enabled=enable, dtype=dtype)  # type: ignore[attr-defined]
        elif device_type == "cuda":
            self._ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)  # type: ignore[attr-defined]
        else:  # mps
            # Conservative approach: enable autocast only for float16
            if dtype == torch.float16:
                self._ctx = torch.amp.autocast(device_type="mps", dtype=dtype)  # type: ignore[arg-type, attr-defined]
            else:
                self._ctx = nullcontext()

    def __enter__(self):  # type: ignore[override]
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
        return self._ctx.__exit__(exc_type, exc, tb)


def setup(
    device: DeviceKind, dtype: DTypeKind, seed: int
) -> Tuple[str, torch.dtype, AmpContext]:
    """Set seeds, select available device, enable TF32 on CUDA, and return autocast ctx.

    Returns (device_type, ptdtype, ctx)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Enable TF32 for better perf on Ampere+ when using CUDA
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Resolve device availability
    if device == "cuda" and torch.cuda.is_available():
        device_type: DeviceKind = "cuda"
    elif (
        device == "mps"
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        device_type = "mps"
    else:
        device_type = "cpu"

    ptdtype = DTYPE_MAP[dtype]
    ctx = AmpContext(device_type, ptdtype)
    return device_type, ptdtype, ctx
