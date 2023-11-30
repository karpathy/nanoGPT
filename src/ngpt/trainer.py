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
# import logging
from typing import Any, Optional, Union
import time
import math
from pathlib import Path
from dataclasses import asdict
# from enrich.console import get_console

import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
# import logging

from ngpt.model import GPT
from rich.text import Text
from enrich import get_logger
# from ngpt import get_logger
from ezpz.history import BaseHistory
from ezpz import get_rank, get_world_size
from ngpt.configs import ExperimentConfig, ModelConfig, add_to_ckpts_file

from tqdm.auto import trange
# if is_interactive():
#     from tqdm.notebook import trange
# else:
#     from tqdm.auto import trange
# from tqdm.autonotebook import trange
# from tqdm import tqdm
# from tqdm import trange



log = get_logger(__name__, level="INFO")
# log = logging.getLogger(__name__)

RANK = get_rank()
WORLD_SIZE = get_world_size()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ScalarLike = Union[float, int, np.floating, bool]


# class TqdmLoggingHandler(logging.StreamHandler):
#     """Avoid tqdm progress bar interruption by logger's output to console"""
#     # see logging.StreamHandler.eval method:
#     # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
#     # and tqdm.write method:
#     # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620
#
#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             tqdm.write(msg, end=self.terminator)
#         except RecursionError:
#             raise
#         except Exception:
#             self.handleError(record)


# log.addHandler(TqdmLoggingHandler())


def format_pair(k: str, v: ScalarLike) -> str:
    if isinstance(v, (int, bool, np.integer)):
        # return f'{k}={v:<3}'
        return f'{k}={v}'
    # return f'{k}={v:<3.4f}'
    return f'{k}={v:<.3f}'


def summarize_dict(d: dict) -> str:
    return ' '.join([format_pair(k, v) for k, v in d.items()])


def grab_tensor(x: Any) -> np.ndarray | ScalarLike | None:
    if x is None:
        return None
    if isinstance(x, (int, float, bool, np.floating)):
        return x
    if isinstance(x, list):
        if isinstance(x[0], torch.Tensor):
            return grab_tensor(torch.stack(x))
        elif isinstance(x[0], np.ndarray):
            return np.stack(x)
        else:
            import tensorflow as tf
            if isinstance(x[0], tf.Tensor):
                return grab_tensor(tf.stack(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif callable(getattr(x, 'numpy', None)):
        assert callable(getattr(x, 'numpy'))
        return x.numpy()
    raise ValueError


def _average(val):
    if isinstance(val, (list, tuple)):
        if isinstance(val[0], torch.Tensor):
            val = grab_tensor(torch.stack(val))
        elif isinstance(val, np.ndarray):
            val = np.stack(val)
        else:
            val = val
    if isinstance(val, torch.Tensor):
        val = grab_tensor(val)

    if isinstance(val, (float, int, bool, np.floating, np.integer)):
        return val
    try:
        avg = np.mean(val).real  # type: ignore
        assert isinstance(avg, np.floating)
        return avg
    except Exception:
        log.exception(f'Failed to average {val}')
        log.warning('Returning val as is')
        return val


def average_dict(d: dict) -> dict:
    avgs = {}
    avg = 0.0
    for key, val in d.items():
        if val is None:
            continue
        if isinstance(val, dict):
            for k, v in val.items():
                kk = f'{key}/{k}'
                avg = _average(v)
                avgs[kk] = avg
        else:
            avg = _average(val)
            avgs[key] = avg
    return avgs


class Trainer:
    def __init__(self, config: ExperimentConfig):
        # self.console = get_console()
        self.config = config
        self.ckpt = None
        # NOTE: ---------------------------------------------------------
        # config.optimizer.gas = (
        #     1 if config.optimizer.gradient_accumulation_steps is None
        #     else config.optimizer.gradient_accumulation_steps
        # ) -------------------------------------------------------------
        self.train_history = BaseHistory()
        self._gas = self.config.optimizer.gas
        self._lr = self.config.optimizer.learning_rate
        self._min_lr = self.config.optimizer.min_lr
        self._diters = self.config.optimizer.lr_decay_iters
        self._witers = self.config.train.warmup_iters
        if self.config.train.init_from == 'scratch':
            log.info('Initializing a new model from scratch')
            model = GPT(self.config.model)
        elif self.config.train.init_from == 'resume':
            model, ckpt = self.restore_from_ckpt()
            self.ckpt = ckpt
            self.config.set_iter_num(ckpt.get('iter_num', 0))
            self.config.set_best_val_loss(ckpt.get('best_val_loss', 1e9))
        elif self.config.train.init_from.startswith('gpt2'):
            log.info(f'Initializing from OpenAI GPT-2 Weights: {self.config.train.init_from}')
            override_args = {'dropout': self.config.model.dropout}
            model = GPT.from_pretrained(self.config.train.init_from, override_args)
            model_cfg = {}
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_cfg[k] = getattr(model.config, k)
            self.config.reset_model_config(ModelConfig(**model_cfg))
        else:
            raise ValueError(f'Unexpected `init_from` = {self.config.train.init_from}. Exiting!')
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
        assert isinstance(self.model, GPT)
        assert issubclass(GPT, torch.nn.Module)
        num_params = self.model.get_num_params()
        if wandb.run is not None:
            wandb.run.config['num_params'] = num_params
        # model_block_size = int(self.model.config.block_size)
        if self.config.model.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.config.model.block_size)
            self.config.model.set_block_size(self.config.model.block_size)
        # self.model.to(self.config.device)
        self.scaler = GradScaler(enabled=(self.config.train.dtype == 'float16'))
        self.optimizer = self.model.configure_optimizers(
            weight_decay=self.config.optimizer.weight_decay,
            learning_rate=self.config.optimizer.learning_rate,
            betas=(
                self.config.optimizer.beta1,
                self.config.optimizer.beta2,
            ),
            device_type=self.config.device_type,
        )
        if self.config.train.init_from == 'resume':
            assert (
                self.ckpt is not None
                and isinstance(self.ckpt, dict)
                and 'optimizer' in self.ckpt
            )
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            self.ckpt = None  # free up memory
        if self.config.train.compile:
            # unoptimized_model = self.model
            self.model = torch.compile(model)
        # if WORLD_SIZE > 1:
        self.model = DDP(model)  # , device_ids=get_local_rank())

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        # data = self.config.train_data if split == 'train' else self.config.val_data
        data = self.config.data.data.get(split, None)
        assert data is not None
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
            x = x.pin_memory().to(self.config.device_type, non_blocking=True)
            y = y.pin_memory().to(self.config.device_type, non_blocking=True)
        else:
            x = x.to(self.config.device_type)
            y = y.to(self.config.device_type)
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
        for split in self.config.data.data.keys():
            losses = torch.zeros(self.config.train.eval_iters)
            for k in range(self.config.train.eval_iters):
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
        log.info(f'Resuming training from {self.config.data.out_dir}')
        ckpt_dir = str(self.config.data.out_dir) if ckpt_dir is None else ckpt_dir
        assert ckpt_dir is not None
        ckpt_path = Path(ckpt_dir).joinpath('ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.config.train.device)
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
            'dt': np.array(dt),
            'dt_tot': np.sum(dt),
            'dt_avg': np.mean(dt),
            'dtf': np.array(dtf),
            'dtf_tot': np.sum(dtf),
            'dtf_avg': np.mean(dtf),
            'dtb': np.array(dtb),
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

    def save_ckpt(
            self,
            raw_model: Optional[torch.nn.Module | GPT] = None,
            add_to_wandb: bool = False
    ):
        if raw_model is None:
            model = self.model.module  # type:ignore
        else:
            model = raw_model  # type:ignore
        assert model is not None
        assert isinstance(model, GPT)
        assert issubclass(GPT,  torch.nn.Module)
        ckpt = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': asdict(self.config.model),
            'iter_num': self.config.iter_num,
            'best_val_loss': self.config.best_val_loss,
            'config': asdict(self.config),
        }
        # assert (
        #     isinstance(model, GPT)
        #     and issubclass(GPT, torch.nn.Module)
        # )
        # assert raw_model is not None
        ckptfile = Path(os.getcwd()).joinpath('ckpt.pt')
        modelfile = Path(os.getcwd()).joinpath('model.pth')
        log.info(f'Saving checkpoint to: {os.getcwd()}')
        log.info(f'Saving model to: {modelfile}')
        torch.save(model.state_dict(), modelfile.as_posix())
        torch.save(ckpt, ckptfile.as_posix())
        add_to_ckpts_file(Path(os.getcwd()))
        if add_to_wandb and wandb.run is not None:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(modelfile.as_posix())
            wandb.run.log_artifact(artifact)

    def train(
            self,
            train_iters: Optional[int] = None,
    ):
        x, y = self.get_batch('train')
        t0 = time.perf_counter()
        # local_iter_num = 0
        raw_model = self.model.module  #  if WORLD_SIZE > 1 else self.model
        assert isinstance(raw_model, GPT)
        running_mfu = -1.0
        output = {'x': x, 'y': y}
        t0 = time.perf_counter()
        losses = {}
        train_iters = (
            self.config.train.max_iters
            if train_iters is None else train_iters
        )
        for train_iter in trange(
                train_iters,
                disable=(RANK != 0),
                total=train_iters,
        ):
            if self.config.iter_num == 0 and self.config.train.eval_only:
                return
            if self.config.iter_num % self.config.train.eval_interval == 0 and RANK == 0:
                losses = self.estimate_loss()
                if (
                    self.config.iter_num > 0
                    and (losses['val'] < self.config.best_val_loss
                         or self.config.train.always_save_checkpoint)
                ):
                    self.save_ckpt(add_to_wandb=False)
            output = self.train_step(x=output['x'], y=output['y'])
            t1 = time.perf_counter()
            dt = t1 - t0
            tokens_per_sec = self.config.tokens_per_iter / dt
            samples_per_sec = self.config.samples_per_iter / dt
            t0 = t1
            output['timers'] |= {
                'dt_iter': dt,
                'tokens_per_sec': tokens_per_sec,
                'samples_per_sec': samples_per_sec,
            }
            # metrics = output['metrics']
            # metrics |= output['timers']
            lossf = output['metrics']['loss'].item() * self._gas
            output['metrics']['loss_tot'] = lossf
            _ = self.train_history.update(output['timers'])
            _ = self.train_history.update(output['metrics'])
            zero = torch.tensor(0.0)
            if (
                    self.config.iter_num % self.config.train.log_interval == 0
                    and (RANK == 0)
            ):
                if train_iter >= 5:
                    mfu = raw_model.estimate_mfu(
                        (self.config.model.batch_size * self.config.optimizer.gas),
                        dt=dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0
                        else 0.9 * running_mfu + 0.1 * mfu
                    )
                pvars = {
                    'step': self.config.iter_num,
                    'loss': lossf,
                    'dt': dt * 1000,
                    'sps': samples_per_sec,
                    'mtps': tokens_per_sec / int(1e6),
                    'mfu': running_mfu * 100,
                    'train_loss': losses.get('train', zero).item(),
                    'val_loss': losses.get('val', zero).item(),
                }
                summary = summarize_dict(pvars)
                log.info(Text(summary))
                if wandb.run is not None:
                    losses |= {
                        'lossf': lossf,
                        'mfu': running_mfu * 100,
                        'iter': self.config.iter_num,
                    }
                    losses['lossf'] = lossf
                    losses['iter'] = self.config.iter_num
                    # wbmetrics = {
                    #     f'training/{k}': v for k, v in metrics.items()
                    # }
                    wbmetrics = {
                        f'Training/{k}': (
                            (wandb.Histogram(v.tolist())
                                if isinstance(v, np.ndarray) else v)
                        ) for k, v in output['metrics'].items()
                    }
                    wbmetrics |= {
                        f'Timing/{k}': (
                            (wandb.Histogram(v.tolist())
                                if isinstance(v, np.ndarray) else v)
                        ) for k, v in output['timers'].items()
                    }
                    wbmetrics |= {
                        f'Loss/{k}': v for k, v in losses.items()
                    }
                    wandb.run.log(wbmetrics)
                    # wandb.run.log({
                    #     'losses': losses,
                    #     'metrics': output['metrics'],
                    #     'timers': output['timers'],
                    #     # 'training': metrics,
                    # })

    def evaluate(
            self,
            s: str,
            num_samples: int = 10,
            max_new_tokens = 500,
            temperature: float = 0.8,
            top_k: int = 200,
            display: Optional[bool] = True,
    ) -> dict[str, str]:
        # seed: Optional[int] = None,
        assert isinstance(self.model.module, GPT)
        assert issubclass(GPT, torch.nn.Module)
        self.model.eval()
        outputs = {}
        with torch.no_grad():
            start_ids = self.config.data.encode(s)
            x = (torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...])
            for idx in range(num_samples):
                y = self.model.module.generate(
                    x,
                    max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                response = self.config.data.decode(y[0].tolist())
                # outputs.append(response)
                response_ = [i for i in response.split('\n')]
                prompt = response_[0]
                responses = [*response_[1:]]
                ret0 = fr"[prompt]: '{prompt}'"
                ret1 = '> ' + '\n> '.join(responses)
                if display:
                    log.info(f'{ret0}')
                    log.info(f'{ret1}')
                outputs[f'{idx}'] = {
                    'raw': response,
                    'prompt': Text(ret0, style='string'),
                    'formatted': Text(ret1, style='blockquote'),
                }
                # log.info(f'[prompt]: "{s}"')
                # # responses = reponse.split('\n ')
                # log.info('> "' + '\n> '.join(response.split('\n ')) + '"')
                #
                # log.info('\n'.join)
                # log.info(f'> "{response}"')
                # log.info(100 * '-')
        return outputs
