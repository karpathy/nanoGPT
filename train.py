import argparse
from rich import print
import os
import time
import csv
from datetime import datetime
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter

from model import GPTConfig, GPT


def parse_args():
    parser = argparse.ArgumentParser()

    # argparse groups
    model_group = parser.add_argument_group('model_group')
    training_group = parser.add_argument_group('training_group')
    logging_group = parser.add_argument_group('logging_group')

    # I/O args
    training_group.add_argument('--out_dir', default='out', type=str)
    training_group.add_argument('--eval_interval', default=250, type=int)
    training_group.add_argument('--log_interval', default=10, type=int)
    training_group.add_argument('--eval_iters', default=200, type=int)
    training_group.add_argument('--eval_only', action='store_true')

    # Checkpoint args
    training_group.add_argument('--only_save_checkpoint_at_end', action='store_true')
    training_group.add_argument('--always_save_checkpoint', action='store_true')
    training_group.add_argument('--init_from', default='scratch', choices=['scratch', 'resume', 'gpt2*'], type=str)

    # Data args
    training_group.add_argument('--dataset', default='shakespeare_char', type=str)
    training_group.add_argument('--gradient_accumulation_steps', default=1, type=int)
    training_group.add_argument('--batch_size', default=64, type=int)

    # Model args
    model_group.add_argument('--block_size', default=256, type=int)
    model_group.add_argument('--n_layer', default=6, type=int)
    model_group.add_argument('--n_head', default=6, type=int)
    model_group.add_argument('--n_embd', default=384, type=int)
    model_group.add_argument('--dropout', default=0.2, type=float)
    model_group.add_argument('--use_post_ln', default=True, action=argparse.BooleanOptionalAction)

    # NORM VARIATIONS
    model_group.add_argument("--layernorm_variant", type=str, default="rmsnorm", choices=["rmsnorm", "layernorm"])
    model_group.add_argument('--bias', default=False, action=argparse.BooleanOptionalAction, help="only used for layernorm variation option")

    # ACTIVATION VARIATIONS
    model_group.add_argument("--activation_variant", type=str, default="gelu", choices=["relu", "squared_relu"])

    # POSITIONAL EMBEDDING VARIATIONS
    model_group.add_argument('--use_rotary_embeddings', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--rope_variant", type=str, default="rope", choices=["shortrope", "rope"])
    model_group.add_argument("--shortrope_length", type=int, default="16", help="number of embeddings to use with rope, must be <= length, and be even")
    model_group.add_argument('--use_abs_pos_embeddings', default=False, action=argparse.BooleanOptionalAction)

    # SOFTMAX VARIATIONS
    ## Selection of softmax variation for attention and output layers
    model_group.add_argument("--softmax_variant_attn", type=str,
                             default="softmax", choices=["constantmax_quan", "constantmax", "polymax", "strongermax", "softermax", "sigsoftmax", "softmax"])
    model_group.add_argument("--softmax_variant_output", type=str,
                             default="softmax", choices=["constantmax_quan", "constantmax", "polymax", "strongermax", "softermax", "sigsoftmax", "softmax"])

    ## Custom Softmax Variation Options
    model_group.add_argument("--constantmax_initial_beta", type=float, default=0.0)
    model_group.add_argument("--constantmax_initial_gamma", type=float, default=100.0)
    model_group.add_argument('--constantmax_use_euler_base', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--constantmax_base", type=float, default=2.0)

    model_group.add_argument("--polymax_x_intercept", type=float, default=-100.0)
    model_group.add_argument("--polymax_y_intercept", type=float, default=1.0)
    model_group.add_argument("--polymax_power", type=float, default=2.0)
    model_group.add_argument("--polymax_divisor", type=float, default=1000.0)

    model_group.add_argument("--sigsoftmax_use_euler_base", type=float, default=2.0)
    model_group.add_argument("--sigsoftmax_base", type=float, default=2.0)

    model_group.add_argument("--strongermax_strength", type=float, default=2.0)

    # Softermax Specific Options
    model_group.add_argument('--softermax_use_xmax', default=True, action=argparse.BooleanOptionalAction)

    # Optimizer args
    training_group.add_argument('--learning_rate', default=1e-3, type=float)
    training_group.add_argument('--max_iters', default=5000, type=int)
    training_group.add_argument('--weight_decay', default=1e-1, type=float)
    training_group.add_argument('--beta1', default=0.9, type=float)
    training_group.add_argument('--beta2', default=0.99, type=float)
    training_group.add_argument('--grad_clip', default=1.0, type=float)

    # LR schedule args
    training_group.add_argument('--decay_lr', action='store_true')
    training_group.add_argument('--warmup_iters', default=100, type=int)
    training_group.add_argument('--lr_decay_iters', default=5000, type=int)
    training_group.add_argument('--min_lr', default=1e-4, type=float)

    # DDP args
    training_group.add_argument('--backend', default='nccl', type=str)

    # System args
    training_group.add_argument('--device', default='cuda', type=str)
    training_group.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="torch data type for inference, e.g. 'int8'")
    training_group.add_argument('--compile', default=False, action=argparse.BooleanOptionalAction)

    # Logging args
    logging_group.add_argument('--log_project', default='out-test', type=str)
    logging_group.add_argument('--log_run_name', default='logs-test', type=str)

    # CSV logging
    logging_group.add_argument('--csv_log', default=True, action=argparse.BooleanOptionalAction)
    training_group.add_argument('--csv_dir', default='csv_logs', type=str)
    training_group.add_argument('--csv_name', default='output.csv', type=str)

    # Tensorboard args
    logging_group.add_argument('--tensorboard_log', default=True, action=argparse.BooleanOptionalAction)
    logging_group.add_argument('--tensorboard_log_dir', type=str, default='logs')
    logging_group.add_argument('--tensorboard_run_name', type=str, default='logs-test')

    # Wandb args
    logging_group.add_argument('--wandb_log', default=False, action=argparse.BooleanOptionalAction)
    logging_group.add_argument('--wandb_project', type=str, default='out-test')
    logging_group.add_argument('--wandb_run_name', type=str, default='logs-test')

    args = parser.parse_args()
    return args, model_group, training_group, logging_group


class Trainer:
    def __init__(self, args, model_group):
        self.args = args
        self.model_group = model_group
        self.setup()

    def setup(self):
        # Setup DDP
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend=self.args.backend)
            print(self.args)
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            print("this is my device", self.device)
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            self.gradient_accumulation_steps //= self.ddp_world_size
        else:
            self.device = self.args.device
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1

        self.tokens_per_iter = self.args.gradient_accumulation_steps * self.ddp_world_size * self.args.batch_size * self.args.block_size

        if self.master_process:
            os.makedirs(self.args.out_dir, exist_ok=True)

        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device_type = 'cuda' if 'cuda' in self.args.device else 'cpu'
        self.ptdtype = {"bfloat16" : torch.bfloat16, "float16" : torch.float16, "float32" : torch.float32}[self.args.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

        # Data loader
        self.train_data = np.memmap(os.path.join('data', self.args.dataset, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_data = np.memmap(os.path.join('data', self.args.dataset, 'val.bin'), dtype=np.uint16, mode='r')
        meta_path = os.path.join('data', self.args.dataset, 'meta.pkl')
        self.meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.meta_vocab_size = meta['vocab_size']

        # Model
        # TODO only add if they are defined from the argparse
        self.model_args = {action.dest: getattr(self.args, action.dest) for action in self.model_group._group_actions}
        print(self.model_args)
        self.model_args['vocab_size'] = None

        if self.args.init_from == 'scratch':
            self.model_args['vocab_size'] = self.meta_vocab_size if self.meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)
            self.iter_num = 0 # for starting from scratch
            self.best_val_loss = 1e9 # really big number
        elif self.args.init_from == 'resume':
            ckpt_path = os.path.join(self.args.out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            checkpoint_model_args = checkpoint['model_args']
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                self.model_args[k] = checkpoint_model_args[k]
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)
            state_dict = checkpoint['model']
            for k,v in list(state_dict.items()):
                if k.startswith('_orig_mod.'):
                    state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
            self.iter_num = checkpoint['iter_num']
            self.best_val_loss = checkpoint['best_val_loss']
        elif self.args.init_from.startswith('gpt2'):
            override_args = dict(dropout=self.args.dropout)
            self.model = GPT.from_pretrained(self.args.init_from, override_args)
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                self.model_args[k] = getattr(self.model.config, k)

        if self.args.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.args.block_size)
            self.model_args['block_size'] = self.args.block_size

        self.model.to(self.device)

        # Optimizer
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.args.dtype == 'float16'))
        self.optimizer = self.model.configure_optimizers(self.args.weight_decay, self.args.learning_rate,
                                                         (self.args.beta1, self.args.beta2), self.device_type)

        if self.args.compile:
            print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        self.raw_model = self.model.module if self.ddp else self.model

        timestamp_prefix = time.strftime("%Y%m%d-%H%M%S")

        # Tensorboard
        if self.args.tensorboard_log:
            timestamped_run_name = timestamp_prefix + "_" + self.args.tensorboard_run_name
            if self.args.csv_log:
                self.args.csv_name = timestamped_run_name
            log_subpath = os.path.join(self.args.tensorboard_log_dir, timestamped_run_name)
            self.writer = SummaryWriter(log_subpath)

        # Wandb
        if self.args.wandb_log and self.master_process:
            import wandb
            self.args.csv_name = wandb_run_name
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_run_name, config=self.args)

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.args.block_size, (self.args.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.args.block_size]).astype(np.int64)) for i in ix])
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.args.eval_iters)
            for k in range(self.args.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def get_lr(self, it):
        if it < self.args.warmup_iters:
            return self.args.learning_rate * it / self.args.warmup_iters
        if it > self.args.lr_decay_iters:
            return self.args.min_lr
        decay_ratio = (it - self.args.warmup_iters) / (self.args.lr_decay_iters - self.args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.args.min_lr + coeff * (self.args.learning_rate - self.args.min_lr)

    def log_metrics(self, losses, lr, running_mfu, iter_num):
        if self.args.tensorboard_log:
            self.writer.add_scalars(
                "loss", { "train": losses['train'], "val": losses['val'] }, iter_num
            )
            self.writer.add_scalar("mfu_pct", running_mfu * 100, iter_num)
            self.writer.add_scalar("lr", lr, iter_num)

        if self.args.wandb_log and self.master_process:
            import wandb
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })

        if self.args.csv_log:
            self.write_to_csv(losses['train'].item(), losses['val'].item())

    def write_to_csv(self, *args):
        os.makedirs(self.args.csv_dir, exist_ok=True)
        csv_path = os.path.join(self.args.csv_dir, self.args.csv_name)
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            # Write arguments as a new row in the CSV
            writer.writerow(args)


    def log_metrics_non_validation(self, loss_training, running_mfu, iter_num):
        if self.args.tensorboard_log:
            self.writer.add_scalars(
                "loss", { "train": loss_training }, iter_num
            )
            self.writer.add_scalar("mfu_pct", running_mfu * 100, iter_num)

        if self.args.wandb_log and self.master_process:
            import wandb
            wandb.log({
                "iter": iter_num,
                "train/loss": loss_training,
                "mfu": running_mfu*100,
            })

    def train(self):
        self.X, self.Y = self.get_batch('train')
        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0

        while True:
            lr = self.get_lr(self.iter_num) if self.args.decay_lr else self.args.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if self.iter_num % self.args.eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                self.log_metrics(losses, lr, running_mfu, self.iter_num)

                if losses['val'] < self.best_val_loss or self.args.always_save_checkpoint:
                    self.best_val_loss = losses['val']
                    if self.iter_num > 0:
                        checkpoint = {
                            'model': self.raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': self.iter_num,
                            'best_val_loss': self.best_val_loss,
                            'config': vars(self.args),
                        }
                        print(f"saving checkpoint to {self.args.out_dir}")
                        torch.save(checkpoint, os.path.join(self.args.out_dir, 'ckpt.pt'))

            if self.iter_num == 0 and self.args.eval_only:
                break

            for micro_step in range(self.args.gradient_accumulation_steps):
                if self.ddp:
                    self.model.require_backward_grad_sync = (micro_step == self.args.gradient_accumulation_steps - 1)

                with self.ctx:
                    logits, loss = self.model(self.X, self.Y)
                    loss = loss / self.args.gradient_accumulation_steps

                self.X, self.Y = self.get_batch('train')

                self.scaler.scale(loss).backward()

            if self.args.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % self.args.log_interval == 0 and self.master_process:
                lossf = loss.item() * self.args.gradient_accumulation_steps
                if local_iter_num >= 5:
                    mfu = self.raw_model.estimate_mfu(self.args.batch_size * self.args.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f} ms, mfu {running_mfu*100:.2f}%")
                self.log_metrics_non_validation(lossf, running_mfu, self.iter_num)

            self.iter_num += 1
            local_iter_num += 1

            if self.iter_num > self.args.max_iters:
                if self.args.only_save_checkpoint_at_end:
                    checkpoint = {
                        'model': self.raw_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'model_args': self.model_args,
                        'iter_num': self.iter_num,
                        'best_val_loss': self.best_val_loss,
                        'config': self.args,
                    }
                    print(f"saving checkpoint to {self.args.out_dir}")
                    torch.save(checkpoint, os.path.join(self.args.out_dir, 'ckpt.pt'))
                break

        if self.args.tensorboard_log:
            self.writer.flush()
            self.writer.close()

        if self.args.wandb_log and self.master_process:
            import wandb
            wandb.log({"finished": True})
            wandb.finish()

def main():
    args, model_group, _, _ = parse_args()
    trainer = Trainer(args, model_group)
    trainer.train()

    if trainer.ddp:
        destroy_process_group()

    if args.tensorboard_log:
        trainer.writer.flush()
        trainer.writer.close()

if __name__ == '__main__':
    main()

