import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from trainers.base import BaseTrainer

class DefaultTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, val_loader):
        super().__init__(config, model, train_loader, val_loader)
        self.optimizer = None
        self.scaler = None
        self.ctx = None

    def _setup(self):
        # DDP setup
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend=self.config['trainer']['backend'])
            ddp_rank = int(os.environ['RANK'])
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(device)
            self.master_process = ddp_rank == 0
            self.seed_offset = ddp_rank
            self.gradient_accumulation_steps = self.config['trainer']['gradient_accumulation_steps']
            assert self.gradient_accumulation_steps % ddp_world_size == 0
            self.gradient_accumulation_steps //= ddp_world_size
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            self.gradient_accumulation_steps = self.config['trainer']['gradient_accumulation_steps']

        if self.master_process:
            os.makedirs(self.config['out_dir'], exist_ok=True)

        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_type = 'cuda' if 'cuda' in self.config['trainer']['device'] else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config['trainer']['dtype']]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # Optimizer
        self.optimizer = self.model.configure_optimizers(
            float(self.config['trainer']['weight_decay']),
            float(self.config['trainer']['learning_rate']),
            (float(self.config['trainer']['beta1']), float(self.config['trainer']['beta2'])),
            device_type
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config['trainer']['dtype'] == 'float16'))

        # Compile model
        if self.config['trainer']['compile']:
            print("Compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model)

        # Wrap model in DDP
        if self.ddp:
            self.model = DDP(self.model, device_ids=[ddp_local_rank])

    def _get_lr(self, it):
        learning_rate = float(self.config['trainer']['learning_rate'])
        warmup_iters = self.config['trainer']['warmup_iters']
        lr_decay_iters = self.config['trainer']['lr_decay_iters']
        min_lr = float(self.config['trainer']['min_lr'])

        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    @torch.no_grad()
    def _estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config['trainer']['eval_iters'])
            data_loader = self.train_loader if split == 'train' else self.val_loader
            data_iter = iter(data_loader)
            for k in range(self.config['trainer']['eval_iters']):
                try:
                    X, Y = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    X, Y = next(data_iter)

                # Move data to the correct device
                X = X.to(self.config['trainer']['device'])
                Y = Y.to(self.config['trainer']['device'])

                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        self._setup()

        train_iter = iter(self.train_loader)
        t0 = time.time()
        local_iter_num = 0
        raw_model = self.model.module if self.ddp else self.model
        running_mfu = -1.0
        iter_num = 0
        best_val_loss = 1e9

        while True:
            lr = self._get_lr(iter_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if iter_num % self.config['trainer']['eval_interval'] == 0 and self.master_process:
                losses = self._estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                if self.config.get('wandb_log') and self.master_process:
                    import wandb
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    })

                if losses['val'] < best_val_loss or self.config.get('always_save_checkpoint', False):
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': raw_model.config,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': self.config,
                        }
                        print(f"saving checkpoint to {self.config['out_dir']}")
                        torch.save(checkpoint, os.path.join(self.config['out_dir'], 'ckpt.pt'))

            if iter_num == 0 and self.config['trainer'].get('eval_only', False):
                break

            for micro_step in range(self.gradient_accumulation_steps):
                try:
                    X, Y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    X, Y = next(train_iter)

                # Move data to the correct device
                X = X.to(self.config['trainer']['device'])
                Y = Y.to(self.config['trainer']['device'])

                if self.ddp:
                    self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

            if self.config['trainer']['grad_clip'] != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['trainer']['grad_clip'])

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.config['trainer']['log_interval'] == 0 and self.master_process:
                lossf = loss.item() * self.gradient_accumulation_steps
                if local_iter_num >= 5:
                    mfu = raw_model.estimate_mfu(self.config['trainer']['batch_size'] * self.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

            iter_num += 1
            local_iter_num += 1

            if iter_num > self.config['trainer']['max_iters']:
                break

        if self.ddp:
            destroy_process_group()
