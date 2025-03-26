
import torch
from contextlib import nullcontext

# Determine the available device
if torch.cuda.is_available():
	device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'
else:
	device = 'cpu'

def is_cuda(dev: str) -> bool:
    return dev.startswith('cuda')

# Set data type based on device
if device == 'cuda' and torch.cuda.is_bf16_supported():
	dtype = torch.bfloat16
else:
	dtype = torch.float16

# Enable compilation if supported
torchCompile = hasattr(torch, 'compile') and (device == 'cuda')


def init_context(dev, dtype):
    if is_cuda(dev):
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)
    else:
        ctx = nullcontext()
    return ctx
