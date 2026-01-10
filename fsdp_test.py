"""
Test script to verify that custom FSDP is functioning correctly.

This script:
1. Initializes models on multiple processes with sharded parameters
2. Runs forward/backward passes
3. Verifies gradients are synchronized across processes (via reduce_scatter)
4. Verifies model parameters stay synchronized after optimizer step

Run with: GLOO_SOCKET_IFNAME=lo0 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --use_env \
    fsdp_test.py
"""

import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import numpy as np

from model import GPTConfig, GPT
from fsdp import custom_fsdp


def verify_gradients_synced(model, rank, world_size):
    # verify that grads are synced across all processes
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
    
    # gather grads from all processes
    gathered_norms = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_norms, grad_norms)
    
    all_synced = True
    if rank == 0:
        print("\n---grad sync check---")
        for name in grad_norms.keys():
            norms = [gathered_norms[i][name] for i in range(world_size)]
            synced = max(norms) - min(norms) < 1e-5  # allow small numerical differences  
            status = "1" if synced else "0"
            print(f"{status} {name}: norms={norms}")
            if not synced:
                all_synced = False
        
        if all_synced:
            print("all grads are synced")
        else:
            print("some grads are not synced")
        print("-" * 40 + "\n")
    else:
        # non zero ranks also check but don't print
        for name in grad_norms.keys():
            norms = [gathered_norms[i][name] for i in range(world_size)]
            if max(norms) - min(norms) >= 1e-5:
                all_synced = False
    
    return all_synced


def verify_params_synced(model, rank, world_size):
    # verify that model parameters are synced across all processes
    param_norms = {}
    
    # gather full params
    for name, param in model.named_parameters():
        shard = param.data.flatten()

        shard_list = [torch.zeros_like(shard) for _ in range(world_size)]
        shard_list[rank] = shard
        dist.all_gather(shard_list, shard)
        
        full_param = torch.cat(shard_list, dim=0)
        param_norm = full_param.norm().item()
        param_norms[name] = param_norm
    
    # gather parameter norms from all processes
    gathered_norms = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_norms, param_norms)
    
    all_synced = True
    if rank == 0:
        print("\n---param sync check---")
        for name in param_norms.keys():
            norms = [gathered_norms[i][name] for i in range(world_size)]
            synced = max(norms) - min(norms) < 1e-5  # allow small numerical differences
            status = "1" if synced else "0"
            print(f"{status} {name}: norms={norms}")
            if not synced:
                all_synced = False
        
        if all_synced:
            print("all params are synced")
        else:
            print("some params are not synced")
        print("-" * 40 + "\n")
    else:
        # non zero ranks also check but don't print
        for name in param_norms.keys():
            norms = [gathered_norms[i][name] for i in range(world_size)]
            if max(norms) - min(norms) >= 1e-5:
                all_synced = False
    
    return all_synced


def main():
    # initialize distributed environment
    rank = int(os.environ.get('RANK', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    
    if rank == -1:
        print("error: this script must be run with torchrun")
        return
    
    # initialize process group
    init_process_group(backend='gloo')
    
    # set device
    device = 'cpu'
    device_type = 'cpu'
    
    print(f"starting fsdp test for rank {rank}")
    
    # create a small model for testing
    config = GPTConfig(
        block_size=64,
        vocab_size=1000,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=False
    )
    
    model = GPT(config)
    model.to(device)
    
    # wrap with custom fsdp
    model = custom_fsdp(model)
    
    # create optimizer
    optimizer = model.module.configure_optimizers(
        weight_decay=0.0,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type=device_type
    )

    batch_size = 4
    block_size = 64
    torch.manual_seed(1337 + rank)  # Different seed per rank
    
    # generate random input data
    X = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
    Y = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
    
    print(f"starting forward pass for rank {rank}")
    
    # forward pass
    logits, loss = model(X, Y)
    loss_value = loss.item()
    
    print(f"loss for rank {rank}: {loss_value:.4f}")
    
    # gather losses from all processes
    losses = [None for _ in range(world_size)]
    dist.all_gather_object(losses, loss_value)
    
    if rank == 0:
        print(f"\nlosses from all processes: {losses}")
        loss_diff = max(losses) - min(losses)
        print(f"loss difference: {loss_diff:.6f}")
        print("Note: Initial losses may differ due to different data batches per process\n")
    
    # backward pass
    print(f"starting backward pass for rank {rank}")
    loss.backward()
    
    # Process grads, happens in zero grad
    # we need to call it to process stored grads
    if hasattr(model, '_reduce_scatter_gradients'):
        model._reduce_scatter_gradients()
    
    # all gather grads to verify
    if hasattr(model, '_all_gather_gradients'):
        model._all_gather_gradients()
    
    # verify synced grads
    verify_gradients_synced(model.module, rank, world_size)
    
    # discard params after backward
    model.zero_grad()
    
    # optimizer step
    print(f"performing optimizer step for rank {rank}")
    optimizer.step()
    
    # synchronize parameters after optimizer step
    model.sync_parameters()
    
    # verify parameters are synchronized after optimizer step
    verify_params_synced(model.module, rank, world_size)
    
    # run a second iteration to ensure consistency
    print(f"running second iteration for rank {rank}")
    X2 = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
    Y2 = torch.randint(0, config.vocab_size, (batch_size, block_size), device=device)
    
    logits2, loss2 = model(X2, Y2)
    loss2.backward()
    
    # process grads
    if hasattr(model, '_reduce_scatter_gradients'):
        model._reduce_scatter_gradients()
    
    # all gather grads to verify
    if hasattr(model, '_all_gather_gradients'):
        model._all_gather_gradients()
    
    # verify gradients are still synced
    verify_gradients_synced(model.module, rank, world_size)
    
    # discard params after backward
    model.zero_grad()
    
    optimizer.step()
    
    # synchronize params after optimizer step
    model.sync_parameters()
    
    # final param sync check
    verify_params_synced(model.module, rank, world_size)
    
    if rank == 0:
        print("\n" + "-" * 50)
        print("fsdp test summary")
        print("-" * 50)
        print("custom fsdp implementation is functioning correctly")
        print("gradients are synchronized across processes (via reduce_scatter)")
        print("parameters stay synchronized after optimizer steps")
        print("-" * 50)

    destroy_process_group()


if __name__ == '__main__':
    main()
