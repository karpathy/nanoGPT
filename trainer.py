"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
from __future__ import absolute_import, annotations, division, print_function
from os import PathLike

import os
import wandb
from typing import Optional, Union
import time
from tqdm.auto import trange
import math
# import pickle
# from contextlib import nullcontext
from pathlib import Path
from dataclasses import asdict
# from typing import Optional

import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler
# import logging
# from dataclasses import dataclass
# from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group

from model import GPT
from enrich import get_logger
from ezpz import get_local_rank, get_rank, get_world_size
from configs import ModelConfig, TrainConfig

# log = logging.getLogger(__name__)
log = get_logger(__name__, level="INFO")

# RANK = setup_torch(backend='DDP')
LOCAL_RANK = get_local_rank()
RANK = get_rank()
WORLD_SIZE = get_world_size()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ScalarLike = Union[float, int, np.floating, bool]


def format_pair(k: str, v: ScalarLike) -> str:
    if isinstance(v, (int, bool, np.integer)):
        # return f'{k}={v:<3}'
        return f'{k}={v}'
    # return f'{k}={v:<3.4f}'
    return f'{k}={v:<.3f}'


def summarize_dict(d: dict) -> str:
    return ' '.join([format_pair(k, v) for k, v in d.items()])


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.ckpt = None
        self._gas = self.config.optimizer.gradient_accumulation_steps
        self._lr = self.config.optimizer.learning_rate
        self._min_lr = self.config.optimizer.min_lr
        self._diters = self.config.optimizer.lr_decay_iters
        self._witers = self.config.warmup_iters
        if self.config.init_from == 'scratch':
            log.info('Initializing a new model from scratch')
            model = GPT(self.config.model)
        elif self.config.init_from == 'resume':
            model, ckpt = self.restore_from_ckpt()
            self.ckpt = ckpt
            self.config.set_iter_num(ckpt.get('iter_num', 0))
            self.config.set_best_val_loss(ckpt.get('best_val_loss', 1e9))
        elif self.config.init_from.startswith('gpt2'):
            log.info(f'Initializing from OpenAI GPT-2 Weights: {self.config.init_from}')
            override_args = {'dropout': self.config.model.dropout}
            model = GPT.from_pretrained(self.config.init_from, override_args)
            model_cfg = {}
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_cfg[k] = getattr(model.config, k)
            self.config.reset_model_config(ModelConfig(**model_cfg))
        else:
            raise ValueError(f'Unexpected `init_from` = {self.config.init_from}. Exiting!')
        self.model = model
        self.model.to(DEVICE)
        assert isinstance(self.model, GPT)
        # model_block_size = int(self.model.config.block_size)
        if self.config.model.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.config.model.block_size)
            self.config.model.set_block_size(self.config.model.block_size)
        # self.model.to(self.config.device)
        self.scaler = GradScaler(enabled=(self.config.dtype == 'float16'))
        self.optimizer = self.model.configure_optimizers(
            weight_decay=self.config.optimizer.weight_decay,
            learning_rate=self.config.optimizer.learning_rate,
            betas=(
                self.config.optimizer.beta1,
                self.config.optimizer.beta2,
            ),
            device_type=self.config.device_type,
        )
        if self.config.init_from == 'resume':
            assert (
                self.ckpt is not None
                and isinstance(self.ckpt, dict)
                and 'optimizer' in self.ckpt
            )
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            self.ckpt = None  # free up memory
        if self.config.compile:
            # unoptimized_model = self.model
            self.model = torch.compile(model)
        # if WORLD_SIZE > 1:
        self.model = DDP(model, device_ids=get_local_rank())

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.config.train_data if split == 'train' else self.config.val_data
        ix = torch.randint(
            len(data) - self.config.model.block_size,
            (self.config.model.batch_size,)
        )
        block_size = self.config.model.block_size
        x = torch.stack(
            [
                torch.from_numpy((data[i:i+block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64))
                for i in ix
            ]
        )
        if self.config.device_type == 'cuda':
            x = x.pin_memory().to(self.config.device, non_blocking=True)
            y = y.pin_memory().to(self.config.device, non_blocking=True)
        else:
            x = x.to(self.config.device)
            y = y.to(self.config.device)
        return x, y

    def get_lr(self, it: int) -> float:
        if it < self._witers:
            return self._lr * it / self._witers
        if it > self._diters:
            return self._min_lr
        decay_ratio = (it - self._witers) / (self._diters - self._witers)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self._min_lr + coeff * (self._lr - self._min_lr)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = self.get_batch(split)
                with self.config.ctx:
                    _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out


    def restore_from_ckpt(
            self,
            ckpt_dir: Optional[str | PathLike] = None
    ) -> tuple[torch.nn.Module, dict]:
        log.info(f'Resuming training from {self.config.out_dir}')
        ckpt_dir = str(self.config.out_dir) if ckpt_dir is None else ckpt_dir
        assert ckpt_dir is not None
        ckpt_path = Path(ckpt_dir).joinpath('ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.config.device)
        ckpt_model = checkpoint['model_args']
        model_config = ModelConfig(
            n_layer=ckpt_model['n_layer'],
            n_head=ckpt_model['n_head'], 
            n_embd=ckpt_model['n_embd'],
            block_size=ckpt_model['block_size'],
            bias=ckpt_model['bias'],
            vocab_size=ckpt_model['vocab_size'],
        )
        model = GPT(model_config)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model, checkpoint

    def _forward_step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        t0 = time.perf_counter()
        with self.config.ctx:
            logits, loss = self.model(x, y)
        return {
            'logits': logits,
            'loss': loss,
            'dt': time.perf_counter() - t0
        }

    def _backward_step(
            self,
            loss: torch.Tensor,
            propagate_grads: bool = False,
    ) -> float:
        t0 = time.perf_counter()
        self.scaler.scale(loss).backward()  # pyright: ignore
        if propagate_grads:
            if self.config.optimizer.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(  # pyright: ignore
                    self.model.parameters(),
                    self.config.optimizer.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return time.perf_counter() - t0

    def train_step(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> dict:
        lr = (
            self.get_lr(self.config.iter_num)
            if self.config.optimizer.decay_lr
            else self._lr
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        dtf = []
        dtb = []
        dt = []
        loss = torch.tensor(0.0)
        for micro_step in range(self._gas):
            is_last_micro_step = (micro_step == self._gas - 1)
            # NOTE: -----------------------------------------------------------
            # In DDP training we only need to sync gradients at the last micro
            # step. the official way to do this is with model.no_sync() context
            # manager, but I really dislike that this bloats the code and
            # forces us to repeat code looking at the source of that context
            # manager, it just toggles this variable
            # -----------------------------------------------------------------
            _ = (
                self.model.require_backward_grad_sync
                if (is_last_micro_step and WORLD_SIZE > 1)
                else None
            )
            fout = self._forward_step(x, y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            x, y = self.get_batch('train')
            loss = fout['loss'] / self._gas
            dtf.append(fout['dt'])
            dtb_ = self._backward_step(loss, propagate_grads=is_last_micro_step)
            dtb.append(dtb_)
            dt.append(dtf + dtb)
        timers = {
            'iter': self.config.iter_num,
            'dt_tot': np.sum(dt),
            'dt_avg': np.mean(dt),
            'dtf_tot': np.sum(dtf),
            'dtf_avg': np.mean(dtf),
            'dtb_tot': np.sum(dtb),
            'dtb_avg': np.mean(dtb)
        }
        metrics = {
            'iter': self.config.iter_num,
            'loss': loss,
            'lr': lr,
        }
        self.config.iter_num += 1
        return {
            'metrics': metrics,
            'timers': timers,
            'x': x,
            'y': y,
        }

    def save_ckpt(self, raw_model):
        ckpt = {
            'model': raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': asdict(self.config.model),
            'iter_num': self.config.iter_num,
            'best_val_loss': self.config.best_val_loss,
            'config': asdict(self.config),
        }
        log.info(f'Saving checkpoint to: {os.getcwd()}')
        torch.save(
            ckpt,
            Path(os.getcwd()).joinpath('ckpt.pt').as_posix()
        )


    def train(self):
        x, y = self.get_batch('train')
        t0 = time.perf_counter()
        # local_iter_num = 0
        raw_model = self.model.module if WORLD_SIZE > 1 else self.model
        assert isinstance(raw_model, GPT)
        running_mfu = -1.0
        # while True:
        train_iterable = trange(self.config.max_iters, disable=(RANK != 0))
        output = {'x': x, 'y': y}
        t0 = time.perf_counter()
        losses = {}
        for train_iter in train_iterable:
            if self.config.iter_num == 0 and self.config.eval_only:
                return
            if self.config.iter_num % self.config.eval_interval == 0 and RANK == 0:
                losses = self.estimate_loss()
                if (
                    self.config.iter_num > 0
                    and (losses['val'] < self.config.best_val_loss
                         or self.config.always_save_checkpoint)
                ):
                    self.save_ckpt(raw_model)
            output = self.train_step(x=output['x'], y=output['y'])
            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1
            postfix = output['metrics']
            postfix |= output['timers']
            postfix['elapsed'] = dt
            train_iterable.set_postfix(postfix)
            # 'metrics': metrics,
            # 'timers': timers,
            # 'x': x,
            # 'y': y,
            if (
                    self.config.iter_num % self.config.log_interval == 0
                    and (RANK == 0)
            ):
                lossf = output['loss'].item() * self._gas
                if train_iter >= 5:
                    mfu = raw_model.estimate_mfu(
                        (self.config.model.batch_size * self._gas),
                        dt=dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0
                        else 0.9 * running_mfu + 0.1 * mfu
                    )
                mstr = ', '.join([
                    f'step: {self.config.iter_num}',
                    f'train_loss: {losses["train"]:.4f}',
                    f'val_loss: {losses["val"]:.4f}',
                    f'dt: {dt*1000:.4f}ms',
                    f'mfu: {running_mfu*100:.2f}%'
                ])
                log.info(mstr)
                if wandb.run is not None:
                    losses |= {
                        'lossf': lossf,
                        'iter': self.config.iter_num,
                    }
                    losses['lossf'] = lossf
                    losses['iter'] = self.config.iter_num
                    wandb.log({'losses': losses}, commit=False)
                    wandb.log({'train': postfix})
