import torch.distributed as dist

class custom_ddp:
    
    def __init__(self, module):
        self.module = module
        self.require_backward_grad_sync = True
        
      
        if not dist.is_initialized():
            raise RuntimeError("custom_ddp requires torch.distributed to be initialized")
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self._register_hooks()
        
    def _register_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_backward_hook())
    
    def _make_backward_hook(self):
        
        def hook(grad):
            if self.require_backward_grad_sync and grad is not None:
                
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad.div_(self.world_size)
            return grad
        return hook
    
    def sync_parameters(self):
        
        for param in self.module.parameters():
            if param.requires_grad:
              
                dist.broadcast(param.data, src=0)
    
    def forward(self, *args, **kwargs):
        # forward pass through the wrapped module
        return self.module(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        # make the wrapper callable, delegating to the module's forward
        return self.module(*args, **kwargs)
    
    def __getattr__(self, name):
        # delegate attribute access to the wrapped module
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
    def state_dict(self, *args, **kwargs):
        # get state dict from the wrapped module
        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        # load state dict into the wrapped module
        return self.module.load_state_dict(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        # get parameters from the wrapped module.
        return self.module.parameters(*args, **kwargs)
    
    def named_parameters(self, *args, **kwargs):
        # get named parameters from the wrapped module
        return self.module.named_parameters(*args, **kwargs)
    
    def train(self, mode=True):
        # set training mode
        return self.module.train(mode)
    
    def eval(self):
        # set evaluation mode
        return self.module.eval()
    
    def zero_grad(self, *args, **kwargs):
        # zero gradients
        return self.module.zero_grad(*args, **kwargs)