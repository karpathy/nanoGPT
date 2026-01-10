import torch
import torch.distributed as dist


class custom_fsdp:
    """
    fsdp implementation

    - Constructor: shard parameters, each rank keeps its own shard
    - Forward: all gather shards -> forward -> discard non-local shards
    - Backward: all gather shards -> backward -> reduce scatter grads -> discard
    """
    
    def __init__(self, module):
        self.module = module
        self.require_backward_grad_sync = True
        
        if not dist.is_initialized():
            raise RuntimeError("custom_fsdp requires torch.distributed")
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # shard the model parameters, each rank only keep its shard
        self._shard_parameters()
        
        # register backward hooks for reduce scatter
        self._register_backward_hooks()
        
    def _shard_parameters(self):
        # shard all params, only rank keeps it's relevant data
        self.param_shapes = {}
        self.param_shard_info = {}
        
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                # store original shape of the parameter
                shape = param.shape
                self.param_shapes[name] = shape
                numel = param.numel()  # total number of elements in the parameter
                
                # calculate the shard size
                shard_size = (numel + self.world_size - 1) // self.world_size
                start_idx = self.rank * shard_size
                end_idx = min(start_idx + shard_size, numel)
                
                # flatten parameter and extract this rank's shard
                param_flat = param.data.flatten()
                shard = param_flat[start_idx:end_idx].clone()
                
                # store shard info
                self.param_shard_info[name] = {
                    'shard_size': shard_size,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'numel': numel
                }
                
                # create a tensor with the original shape but only populate the shard portion
                # preserves the parameter shape
                shard_tensor = torch.zeros(numel, device=param.data.device, dtype=param.data.dtype)
                shard_tensor[start_idx:end_idx] = shard
                shard_tensor = shard_tensor.view(shape)
                
                # update parameter data in place
                param.data = shard_tensor
    
    def _get_parameter(self, name):
        parts = name.split('.')
        obj = self.module
        for part in parts:
            obj = getattr(obj, part)
        return obj
    
    def _all_gather_parameters(self):
        # gather all shards from all ranks to reconstruct the full parameters
        gathered_params = {}
        
        for name in self.param_shard_info.keys():
            param = self._get_parameter(name)
            shard_info = self.param_shard_info[name]
            numel = shard_info['numel']
            shape = self.param_shapes[name]
            shard_size = shard_info['shard_size']
            
            # get shard from this rank
            shard = param.data.flatten().clone()
            
            # pad shard to shard_size for all_gather
            if shard.numel() < shard_size:
                padding = torch.zeros(
                    shard_size - shard.numel(),
                    device=shard.device,
                    dtype=shard.dtype
                )
                shard = torch.cat([shard, padding])
            
            # create list to gather all shards
            shard_list = [torch.zeros_like(shard) for _ in range(self.world_size)]
            shard_list[self.rank] = shard
            
            # all gather shards from all ranks
            dist.all_gather(shard_list, shard)
            
            # concatenate all shards to reconstruct full parameter
            full_param_flat = torch.cat(shard_list, dim=0)[:numel]
            full_param = full_param_flat.view(shape)
            
            gathered_params[name] = full_param
        
        return gathered_params
    
    def _restore_full_parameters(self, gathered_params):
        # restore parameters for forward/backward pass
        for name, full_param in gathered_params.items():
            param = self._get_parameter(name)
            shard_info = self.param_shard_info[name]
            shape = self.param_shapes[name]
            
            # reshape the parameter back to its full shape
            full_param_reshaped = full_param.view(shape)
            
            # update the param data in place, ensure parameter has same shape
            param.data = full_param_reshaped.to(param.data.device).to(param.data.dtype)
        
        # re-register hooks on the full parameters
        self._register_backward_hooks()
    
    def _discard_non_local_shards(self):
        # After forward/backward pass, discard the all the shards that are not local and only keep this rank's shard
        # update the parameter in place
        # keep original shape but only store shard data
        for name in self.param_shard_info.keys():
            param = self._get_parameter(name)
            shard_info = self.param_shard_info[name]
            shape = self.param_shapes[name]
            
            # get the full parameter data
            param_flat = param.data.flatten()
            
            # if parameter is full extract shard
            if param_flat.numel() == shard_info['numel']:
                # extract this rank's shard
                shard = param_flat[shard_info['start_idx']:shard_info['end_idx']].clone()
                
                # tensor with original shape but only population this the shard portion
                # create zero tensor and store shard in the correct place
                shard_tensor = torch.zeros(shard_info['numel'], device=param.data.device, dtype=param.data.dtype)
                shard_tensor[shard_info['start_idx']:shard_info['end_idx']] = shard
                
                # reshape to original shape
                param.data = shard_tensor.view(shape).contiguous()
            else:
                # ensure the shape of the sharded portion is correct
                if param.data.numel() != shape.numel():
                    # need to pad/reshape
                    param_flat = param.data.flatten()
                    shard_tensor = torch.zeros(shard_info['numel'], device=param.data.device, dtype=param.data.dtype)
                    actual_shard_size = min(len(param_flat), shard_info['end_idx'] - shard_info['start_idx'])
                    shard_tensor[shard_info['start_idx']:shard_info['start_idx'] + actual_shard_size] = param_flat[:actual_shard_size]
                    param.data = shard_tensor.view(shape).contiguous()
        
        # re-register hooks on sharded parameters
        self._register_backward_hooks()
    
    def _register_backward_hooks(self):
        # handle gradient reduce scatter
        for name in self.param_shard_info.keys():
            param = self._get_parameter(name)
            if param.requires_grad:
                # remove old hooks and register new hooks
                param.register_hook(self._make_backward_hook(name))
    
    def _make_backward_hook(self, name):
        # create backward hook that stores gradients for reduce scatter
        def hook(grad):
            if self.require_backward_grad_sync and grad is not None:
                # store the gradient for later processing after backward is completed
                if not hasattr(self, '_stored_grads'):
                    self._stored_grads = {}
                self._stored_grads[name] = grad.clone()
            # return gradient
            return grad
        return hook
    
    def forward(self, *args, **kwargs):
        # all gather -> forward -> keep for backward

        if self.require_backward_grad_sync:
            # gather all parameter shards before forward
            gathered_params = self._all_gather_parameters()
            self._restore_full_parameters(gathered_params)
            self._params_gathered = True
        
        # run forward pass
        output = self.module(*args, **kwargs)
        
        return output
    
    def __call__(self, *args, **kwargs):
        # make the wrapper callable
        return self.forward(*args, **kwargs)
    
    def sync_parameters(self):
        # synchronize param shards via broadcast from rank 0, all shards are updated consistenly

        for name in self.param_shard_info.keys():
            param = self._get_parameter(name)
            if param.requires_grad:
                dist.broadcast(param.data, src=0)
    
    def __getattr__(self, name):
        # attribute access for wrapped model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
    def state_dict(self, *args, **kwargs):
        # gather parameters temporarily
        gathered_params = self._all_gather_parameters()
        was_gathered = getattr(self, '_params_gathered', False)
        if not was_gathered:
            self._restore_full_parameters(gathered_params)
        
        # get state dict
        state = self.module.state_dict(*args, **kwargs)
        
        # restore shards if not already gathered
        if not was_gathered:
            self._discard_non_local_shards()
        
        return state
    
    def load_state_dict(self, *args, **kwargs):
        # load state dict, sharded automatically
        result = self.module.load_state_dict(*args, **kwargs)
        # re-shard after loading
        self._shard_parameters()
        self._register_backward_hooks()
        return result
    
    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)
    
    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)
    
    def train(self, mode=True):
        return self.module.train(mode)
    
    def eval(self):
        return self.module.eval()
    
    def _reduce_scatter_gradients(self):
        # reduce-scatter grads stored during backward pass
        # each rank has different shard of grad afterward
        if not hasattr(self, '_stored_grads') or not self._stored_grads:
            return
        
        for name, grad in self._stored_grads.items():
            if grad is None:
                continue
                
            param = self._get_parameter(name)
            if not param.requires_grad:
                continue
                
            shard_info = self.param_shard_info[name]
            numel = shard_info['numel']
            shard_size = shard_info['shard_size']
            
            # use the stored full grad
            grad_flat = grad.flatten()
            
            # ensure expected grad size is correct
            if grad_flat.numel() != numel:
                if grad_flat.numel() < numel:
                    padding = torch.zeros(
                        numel - grad_flat.numel(),
                        device=grad_flat.device,
                        dtype=grad_flat.dtype
                    )
                    grad_flat = torch.cat([grad_flat, padding])
                else:
                    grad_flat = grad_flat[:numel]
            
            # pad to make divisible by world size for reduce scatter
            padded_size = shard_size * self.world_size
            if grad_flat.numel() < padded_size:
                padding = torch.zeros(
                    padded_size - grad_flat.numel(),
                    device=grad_flat.device,
                    dtype=grad_flat.dtype
                )
                grad_flat = torch.cat([grad_flat, padding])
            
            # split into shards
            grad_shards = grad_flat.chunk(self.world_size)
            
            # sum grads and scatter
            output_shard = torch.zeros_like(grad_shards[0])
            dist.reduce_scatter(output_shard, list(grad_shards), op=dist.ReduceOp.SUM)
            
            # extract actual shard
            actual_shard_size = shard_info['end_idx'] - shard_info['start_idx']
            grad_shard = output_shard[:actual_shard_size]
            
            # store the sharded grad
            # create a full size grad with zeros and place the shard
            grad_full = torch.zeros(numel, device=grad_shard.device, dtype=grad_shard.dtype)
            grad_full[shard_info['start_idx']:shard_info['end_idx']] = grad_shard
            
            # reshape to original parameter shape
            grad_reshaped = grad_full.view(self.param_shapes[name])
            
            # update parameter gradient with the sharded gradient
            param_shape = param.data.shape
            if param_shape == self.param_shapes[name]:
                # parameter is full, set full grad
                param.grad = grad_reshaped.contiguous()
            else:
                # parameter is sharded, extract the shard portion
                param.grad = grad_reshaped.flatten()[shard_info['start_idx']:shard_info['end_idx']].view(param_shape).contiguous()
        
        # clear stored gradients
        self._stored_grads = {}
    
    def _all_gather_gradients(self):
        # get full synced grads
        # all ranks will have same grad after this
        for name in self.param_shard_info.keys():
            param = self._get_parameter(name)
            if not param.requires_grad or param.grad is None:
                continue
            
            shard_info = self.param_shard_info[name]
            numel = shard_info['numel']
            shard_size = shard_info['shard_size']
            
            # get the current gradient
            grad_flat = param.grad.flatten()
            
            # extract the shard portion
            actual_shard_size = shard_info['end_idx'] - shard_info['start_idx']
            if grad_flat.numel() == numel:
                # full gradient, extract shard
                grad_shard = grad_flat[shard_info['start_idx']:shard_info['end_idx']].clone()
            else:
                # already sharded
                grad_shard = grad_flat.clone()
            
            # pad shard to shard size for all gather
            if grad_shard.numel() < shard_size:
                padding = torch.zeros(
                    shard_size - grad_shard.numel(),
                    device=grad_shard.device,
                    dtype=grad_shard.dtype
                )
                grad_shard = torch.cat([grad_shard, padding])
            
            # collect shards from all ranks
            grad_shards = [torch.zeros_like(grad_shard) for _ in range(self.world_size)]
            dist.all_gather(grad_shards, grad_shard)
            
            # concatenate shards to get full grad
            grad_full_flat = torch.cat(grad_shards)[:numel]
            
            # reshape to original parameter shape
            grad_full = grad_full_flat.view(self.param_shapes[name])
            
            # update parameter gradient with full synced grad
            param.grad = grad_full.contiguous()
    
    def zero_grad(self, *args, **kwargs):
        # zero grads and discard param shards after backward, process stored grads first
        if hasattr(self, '_stored_grads') and self._stored_grads:
            self._reduce_scatter_gradients()
        
        # discard param shards after backward
        if getattr(self, '_params_gathered', False):
            self._discard_non_local_shards()
            self._params_gathered = False
        
        return self.module.zero_grad(*args, **kwargs)
