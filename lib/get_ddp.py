try:
    from torch.nn.parallel import DistributedDataParallel as DDP
except ImportError:
    DDP = None

try:
    from torch.distributed import init_process_group, destroy_process_group
except ImportError:
    init_process_group = None
    destroy_process_group = None
