import torch
from lib.get_context import nullcontext

try:
    from torch.cuda.amp import autocast
    autocast_available = True
except ImportError:
    autocast_available = False

def get_autocast_context(device_type, dtype=None):
    if device_type == 'cpu':
        return nullcontext()
    elif autocast_available:
        try:
            # Try new unified API first
            return torch.amp.autocast(device_type=device_type, dtype=dtype)
        except (AttributeError, TypeError):
            # Fall back to old CUDA-only API
            if device_type == 'cuda':
                return autocast()  # old API, no parameters
            else:
                return nullcontext()
    else:
        return nullcontext()
