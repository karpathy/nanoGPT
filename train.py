import argparse
import sys
from rich import print
import os
import time
import csv
from datetime import datetime
import math
import pickle
from contextlib import nullcontext
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
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
    training_group.add_argument('--patience', default=None, type=int, help="if set, will stop training if the number of evaluations since val loss was seen to decrease exceeds 'patience' setting.")
    training_group.add_argument('--init_from', default='scratch', choices=['scratch', 'prev_run', 'resume', 'gpt2*'], type=str)
    training_group.add_argument('--prev_run_ckpt', default='', type=str)
    training_group.add_argument('--csv_ckpt_dir', default='', type=str)

    # Data args
    training_group.add_argument('--dataset', default='shakespeare_char', type=str)
    training_group.add_argument('--batch_size', default=64, type=int)
    training_group.add_argument("--seed", default=1337, type=int)

    # Model args
    model_group.add_argument('--block_size', default=256, type=int)
    model_group.add_argument('--n_layer', default=6, type=int)
    model_group.add_argument('--n_head', default=6, type=int)
    model_group.add_argument('--n_kv_group', default=6, type=int)
    model_group.add_argument('--n_embd', default=384, type=int)
    model_group.add_argument('--dropout', default=0.2, type=float)
    model_group.add_argument('--use_post_ln', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--use_parallel_mlp', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--window_size', default=None, type=int, help="Sliding window size, note this cannot be greater than block size")
    model_group.add_argument('--gate', default=False, action=argparse.BooleanOptionalAction, help="option for gated attention see https://arxiv.org/abs/2306.12929")

    # Shared Parameter Settings
    model_group.add_argument('--shared_mlp_size', default=1, type=int, help="every 'k' contiguous blocks of mlp are shared")
    model_group.add_argument('--shared_mlp_sym', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--shared_attn_size', default=1, type=int, help="every 'k' contiguous blocks of attn are shared")
    model_group.add_argument('--shared_attn_sym', default=False, action=argparse.BooleanOptionalAction, help="symmetrical attention sharing")

    # NORM VARIATIONS
    model_group.add_argument("--norm_variant_attn", type=str, default="rmsnorm", choices=["krmsnorm", "prmsnorm", "rmsnorm", "layernorm"])
    model_group.add_argument("--norm_variant_output", type=str, default="rmsnorm", choices=["krmsnorm", "prmsnorm", "rmsnorm", "layernorm"])
    model_group.add_argument('--bias', default=False, action=argparse.BooleanOptionalAction, help="only used for layernorm variation option")
    model_group.add_argument("--prmsnorm_pct", default=0.0625, type=float, help="percentage (1 being 100 percent) of first entries used for partial rms" )
    model_group.add_argument("--krmsnorm_num", default=10, type=int, help="max number of first entries for partial rms" )

    # ACTIVATION VARIATIONS
    model_group.add_argument(
        "--activation_variant",
        type=str,
        default="gelu",
        choices=[
            "celu",
            "elu",
            "gelu",
            "glu",
            "leaky_relu",
            "mish",
            "prelu",
            "relu6",
            "rrelu",
            "selu",
            "sigmoid",
            "silu",
            "softplus",
            "softsign",
            "squared_relu",
            "tanh",
        ],
    )

    # LINEAR VARIATIONS
    model_group.add_argument(
        "--linear_variant",
        type=str,
        default="linear",
        choices=[
            "linear",
            "bitlinear",
            "bitlinear_1p58",
            "bitlinear_optimized",
        ],
    )

    # POSITIONAL EMBEDDING VARIATIONS
    model_group.add_argument('--use_rotary_embeddings', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--rope_variant", type=str, default="rope", choices=["shortrope", "rope"])
    model_group.add_argument("--shortrope_length", type=int, default="16", help="number of embeddings to use with rope, must be <= length, and be even")
    model_group.add_argument('--use_abs_pos_embeddings', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--use_fire_embeddings', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--shared_fire_embeddings', default=False, action=argparse.BooleanOptionalAction)

    # SOFTMAX VARIATIONS
    ## Selection of softmax variation for attention and output layers
    model_group.add_argument("--softmax_variant_attn", type=str,
                             default="softmax", choices=[
                                                         "saturatingconsmax",
                                                         "consmax",
                                                         "consmax_quan",
                                                         "polymax",
                                                         "polymax_quan",
                                                         "exppolymax",
                                                         "strongermax",
                                                         "softermax",
                                                         "sigsoftmax",
                                                         "softmax",
                                                         "softplus",
                                                         "squareplus",
                                                         "exppolymax",
                                                         ])
    model_group.add_argument("--softmax_variant_output", type=str,
                             default="softmax", choices=[
                                                         "saturatingconsmax",
                                                         "consmax",
                                                         "consmax_quan",
                                                         "polymax",
                                                         "polymax_quan",
                                                         "exppolymax",
                                                         "strongermax",
                                                         "softermax",
                                                         "sigsoftmax",
                                                         "softmax",
                                                         "softplus",
                                                         "squareplus",
                                                         "exppolymax",
                                                         ])

    ## Custom Softmax Variation Options
    ### ConSmax and SaturatingConSmax Options
    model_group.add_argument("--consmax_initial_beta", type=float, default=2.5)
    model_group.add_argument("--consmax_initial_gamma", type=float, default=100.0)
    model_group.add_argument('--consmax_use_euler_base', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--consmax_base", type=float, default=2.0)

    ### Special Options for SaturatingConSmax
    model_group.add_argument("--consmax_saturation", type=float, default=11.0, help="point where we transition from consmax to linear saturatingconsmax, defaults to 11 to approximate e^x sat for fp16")
    model_group.add_argument('--consmax_learnable_beta', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--consmax_learnable_gamma', default=True, action=argparse.BooleanOptionalAction)

    ### Polymax Options
    model_group.add_argument("--polymax_x_intercept", type=float, default=-100.0)
    model_group.add_argument("--polymax_y_intercept", type=float, default=1.0)
    model_group.add_argument("--polymax_power", type=float, default=2.0)
    model_group.add_argument("--polymax_divisor", type=float, default=1000.0)

    ### SigSoftmax Options
    model_group.add_argument('--sigsoftmax_use_euler_base', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--sigsoftmax_base", type=float, default=2.0)

    ### Strongermax Options - Testing Incremental Adjustments to Regular Softmax
    model_group.add_argument("--strongermax_strength", type=float, default=4.0)
    model_group.add_argument('--strongermax_sum_to_1', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--strongermax_divisor", type=float, default=1.0)
    model_group.add_argument('--strongermax_use_xmax', default=True, action=argparse.BooleanOptionalAction)

    ### ExpPolymax Options
    model_group.add_argument('--exppolymax_use_euler_base', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--exppolymax_base", type=float, default="4")
    model_group.add_argument("--exppolymax_y_intercept", type=float, default=1.0)
    model_group.add_argument("--exppolymax_power", type=float, default=2.0)
    model_group.add_argument("--exppolymax_divisor", type=float, default=1000.0)

    ### Softermax Specific Options
    model_group.add_argument('--softermax_use_xmax', default=True, action=argparse.BooleanOptionalAction)

    # Optimizer args
    training_group.add_argument('--max_iters', default=3500, type=int)
    training_group.add_argument('--weight_decay', default=1e-1, type=float)
    training_group.add_argument('--beta1', default=0.9, type=float)
    training_group.add_argument('--beta2', default=0.99, type=float)
    training_group.add_argument('--grad_clip', default=1.0, type=float)

    # LR schedule args
    training_group.add_argument('--learning_rate', default=1e-3, type=float)
    training_group.add_argument('--min_lr', default=1e-4, type=float)
    training_group.add_argument('--decay_lr', default=False, action=argparse.BooleanOptionalAction)
    training_group.add_argument('--lr_decay_iters', default=3500, type=int)
    training_group.add_argument('--lr_decay_match_max_iters', default=True, action=argparse.BooleanOptionalAction)
    training_group.add_argument('--warmup_iters', default=100, type=int)

    # DDP args
    training_group.add_argument('--backend', default='nccl', type=str)
    training_group.add_argument('--gradient_accumulation_steps', default=1, type=int)

    # System args
    training_group.add_argument('--device', default='cuda', type=str)
    training_group.add_argument("--dtype", type=str, default="float16", choices=["bfloat16", "float16", "float32"], help="torch data type for inference, e.g. 'int8'")
    training_group.add_argument('--compile', default=False, action=argparse.BooleanOptionalAction)

    # Logging args
    logging_group.add_argument('--log_project', default='out-test', type=str)
    logging_group.add_argument('--log_run_name', default='logs-test', type=str)
    logging_group.add_argument('--timestamp', default='', type=str)
    logging_group.add_argument('--save_nan_checkpoint', default=False, action=argparse.BooleanOptionalAction)

    # CSV logging
    logging_group.add_argument('--csv_log', default=True, action=argparse.BooleanOptionalAction)
    logging_group.add_argument('--csv_dir', default='csv_logs', type=str)
    logging_group.add_argument('--csv_name', default='output', type=str, help="Output csv basename. Note, the .csv will be automatically appended.")

    # Tensorboard args
    logging_group.add_argument('--tensorboard_log', default=True, action=argparse.BooleanOptionalAction)
    logging_group.add_argument('--tensorboard_log_dir', type=str, default='logs')
    logging_group.add_argument('--tensorboard_run_name', type=str, default='logs-test')

    # Wandb args
    logging_group.add_argument('--wandb_log', default=False, action=argparse.BooleanOptionalAction)
    logging_group.add_argument('--wandb_project', type=str, default='out-test')
    logging_group.add_argument('--wandb_run_name', type=str, default='logs-test')
    logging_group.add_argument('--statistic', choices=[
        'input_mean', 'input_median', 'input_stdev', 'input_max',
        'output_mean', 'output_median', 'output_stdev', 'output_max'
    ], default='input_mean', help='Select the statistic and type to display, example: input_mean, output_max')


    args = parser.parse_args()
    return args, model_group, training_group, logging_group

def initialize_statistics(num_layers, num_heads):
        stats = {
            'mean': [],
            'median': [],
            'stdev': [],
            'max': [],
            'o_mean': [],
            'o_median': [],
            'o_stdev': [],
            'o_max': []
        }
    
        for _ in range(num_layers):
            stats['mean'].append([[] for _ in range(num_heads)])
            stats['median'].append([[] for _ in range(num_heads)])
            stats['stdev'].append([[] for _ in range(num_heads)])
            stats['max'].append([[] for _ in range(num_heads)])
            stats['o_mean'].append([[] for _ in range(num_heads)])
            stats['o_median'].append([[] for _ in range(num_heads)])
            stats['o_stdev'].append([[] for _ in range(num_heads)])
            stats['o_max'].append([[] for _ in range(num_heads)])
        
        return stats


class Trainer:
    
    def __init__(self, args, model_group):
        self.args = args
        self.model_group = model_group

        # typically make the decay iters equal to max_iters
        if self.args.lr_decay_match_max_iters:
            self.args.lr_decay_iters = self.args.max_iters

        self.setup()
        self.stats = initialize_statistics(self.args.n_layer, self.args.n_head)

    def setup(self):
        # Setup DDP
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend=self.args.backend)
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

        print("seed: ", self.args.seed)
        print("seed offset: ", self.seed_offset)
        torch.manual_seed(self.args.seed + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device_type = 'cuda' if 'cuda' in self.args.device else 'cpu'
        self.ptdtype = {"bfloat16" : torch.bfloat16, "float16" : torch.float16, "float32" : torch.float32}[self.args.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

        # Model
        # TODO only add if they are defined from the argparse
        self.model_args = {action.dest: getattr(self.args, action.dest) for action in self.model_group._group_actions}
        self.model_args['vocab_size'] = None

        if self.args.init_from == 'scratch':
            self.model_args['vocab_size'] = self.get_vocab_size_from_meta()
            self.load_data()
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)
            self.iter_num = 0 # for starting from scratch
            self.best_val_loss = 1e9 # really big number
        elif self.args.init_from == 'resume':
            ckpt_path = os.path.join(self.args.out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            checkpoint_model_args = checkpoint['model_args']
            for k in ['n_layer', 'n_head', 'n_kv_group', 'n_embd', 'block_size', 'bias', 'vocab_size', 'window_size', 'gate']:
                self.model_args[k] = checkpoint_model_args[k]
            self.load_data()
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
            for k in ['n_layer', 'n_head', 'n_kv_group', 'n_embd', 'block_size', 'bias', 'vocab_size', 'window_size', 'gate']:
                self.model_args[k] = getattr(self.model.config, k)
            self.load_data()
        elif self.args.init_from == 'prev_run':
            ckpt_path = os.path.join(self.args.prev_run_ckpt, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            checkpoint_model_args = checkpoint['model_args']
            for k in ['n_layer', 'n_head', 'n_kv_group', 'n_embd', 'block_size', 'bias', 'vocab_size', 'window_size', 'gate']:
                self.model_args[k] = checkpoint_model_args[k]
            self.load_data()
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)
            state_dict = checkpoint['model']
            for k,v in list(state_dict.items()):
                if k.startswith('_orig_mod.'):
                    state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
            self.iter_num = 0
            self.best_val_loss = checkpoint['best_val_loss']

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
        if self.args.timestamp:
            timestamp_prefix = self.args.timestamp

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

    def get_vocab_size_from_meta(self):
        # Data loader
        meta_path = os.path.join('data', self.args.dataset, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                if 'vocab_size' in meta:
                    return meta['vocab_size']
                else:
                    sys.exit(f"Error: 'vocab_size' key not found in {meta_path}")
        else:
            sys.exit(f"Error: File not found - {meta_path}")

    def load_data(self):
        if self.model_args['vocab_size'] is None:
            sys.exit("Error: no vocab size specified")
        elif self.model_args['vocab_size'] == 100277:
            # cl100k_base, vocab size 100277, requires np.uint32
            self.train_data = np.memmap(os.path.join('data', self.args.dataset, 'train.bin'), dtype=np.uint32, mode='r')
            self.val_data = np.memmap(os.path.join('data', self.args.dataset, 'val.bin'), dtype=np.uint32, mode='r')
        else:
            # all other tokenations so far require only np.uint16
            self.train_data = np.memmap(os.path.join('data', self.args.dataset, 'train.bin'), dtype=np.uint16, mode='r')
            self.val_data = np.memmap(os.path.join('data', self.args.dataset, 'val.bin'), dtype=np.uint16, mode='r')

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

    def write_to_csv(self, *args, prefix=""):
        csv_full_dir = self.args.csv_dir
        if self.args.csv_ckpt_dir:
            csv_full_dir = f"{self.args.csv_dir}/{self.args.csv_ckpt_dir}"
        else:
            if self.args.tensorboard_log:
                csv_full_dir = f"{self.args.csv_dir}/{self.args.tensorboard_run_name.split('-')[0]}-{self.args.dataset}"
        os.makedirs(csv_full_dir, exist_ok=True)
        csv_path = os.path.join(csv_full_dir, prefix + self.args.csv_name + ".csv")
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            # Write arguments as a new row in the CSV
            writer.writerow(args)


    def log_gamma_beta(self, gamma, beta, iter_num, layer_num):
        if self.args.tensorboard_log:
            self.writer.add_scalar( "gamma_" + str(layer_num), gamma, iter_num)
            self.writer.add_scalar( "beta_" + str(layer_num), beta, iter_num)

        if self.args.wandb_log and self.master_process:
            import wandb
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })

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
    
    def plot_statistics(self):
        parts = self.args.statistic.split('_')
        data_type = parts[0]  # 'input' or 'output'
        stat_type = parts[1]  # 'mean', 'median', 'stdev', 'max'

        # to decide whether to use the input or output statistics
        stat_prefix = 'o_' if data_type == 'output' else ''
        directory_path = 'out/images'
        os.makedirs(directory_path, exist_ok=True)

        # draw the plot
        fig = go.Figure()
        plt.figure(figsize=(10, 6))
        for layer_idx, stats_per_layer in enumerate(self.stats[stat_prefix + stat_type]):
            for head_idx, data in enumerate(stats_per_layer):
                fig.add_trace(go.Scatter(
                    x=list(range(len(data))),
                    y=data,
                    mode='lines',
                    name=f'Layer {layer_idx + 1} Head {head_idx + 1}'
                ))
                plt.plot(data, label=f'Layer {layer_idx + 1} Head {head_idx + 1}')

        # add titles and legend to Plotly
        fig.update_layout(
            title=f'Change in {stat_type.title()} Values for {data_type.capitalize()} During Training',
            xaxis_title='Training Iteration',
            yaxis_title=f'{stat_type.title()} of {data_type.capitalize()} Softmax Inputs',
            legend_title='Head/Layer'
        )
        fig.write_html(f'{directory_path}/{data_type}_{stat_type}_changes_plot.html')

        # add titles and lengend to Matplotlib
        plt.title(f'Change in {stat_type.title()} Values for {data_type.capitalize()} During Training')
        plt.xlabel('Training Iteration')
        plt.ylabel(f'{stat_type.title()} of {data_type.capitalize()}')
        plt.legend(title='Head/Layer')
        plt.grid(True)
        plt.savefig(f'{directory_path}/{data_type}_{stat_type}_changes_plot.png')
        plt.close()


    def train(self):
        self.X, self.Y = self.get_batch('train')
        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0
        num_steps_with_worse_loss = 0

        while True:
            lr = self.get_lr(self.iter_num) if self.args.decay_lr else self.args.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if self.iter_num % self.args.eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                self.log_metrics(losses, lr, running_mfu, self.iter_num)

                if losses['val'] < self.best_val_loss or self.args.always_save_checkpoint:
                    if losses['val'] < self.best_val_loss:
                        self.iter_num_best_val_loss = self.iter_num
                        self.best_val_loss = losses['val']
                        num_steps_with_worse_loss = 0
                    if self.iter_num > 0:
                        checkpoint = {
                            'model': self.raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': self.iter_num,
                            'best_val_loss': self.best_val_loss,
                            'nan_iter_num' : None,
                            'nan' : None,
                            'config': vars(self.args),
                        }
                        print(f"saving checkpoint to {self.args.out_dir}")
                        torch.save(checkpoint, os.path.join(self.args.out_dir, 'ckpt.pt'))
                if self.args.patience is not None and num_steps_with_worse_loss >= self.args.patience:
                    print(f"Early Stopping: loss has not decreased in {self.args.patience + 1} steps")
                    break
                if losses['val'] > self.best_val_loss:
                    num_steps_with_worse_loss += 1

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
                if math.isnan(lossf):
                    if self.args.save_nan_checkpoint:
                        checkpoint = {
                            'model': self.raw_model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'model_args': self.model_args,
                            'iter_num': self.iter_num_best_val_loss,
                            'best_val_loss': self.best_val_loss,
                            'nan_iter_num' : self.iter_num,
                            'nan' : True,
                            'config': vars(self.args),
                        }
                        print(f"saving checkpoint to {self.args.out_dir}")
                        torch.save(checkpoint, os.path.join(self.args.out_dir, 'ckpt.pt'))
                    sys.exit("Exiting training loss is NaN")
                self.log_metrics_non_validation(lossf, running_mfu, self.iter_num)



            if self.args.softmax_variant_attn == "constantmax":
                betas = []
                gammas = []
                i_sum_vals = []
                i_means = []
                i_medians = []
                i_stdevs = []
                i_max_values = []
                denominator = []
                o_sum_vals = []
                o_means = []
                o_medians = []
                o_stdevs = []
                o_max_values = []

                for layer in range (self.args.n_layer):
                    # Inputs
                    inputs_location = f"transformer.h[{layer}].attn.softmax_layer.inputs"
                    
                    softmax_input = eval(f"self.model.{inputs_location}").to('cpu').to(torch.float32)
                    

                    ## Get first batch
                    i_first_batch = softmax_input[0]
                    i_first_batch[i_first_batch == float('-inf')] = float('NaN')


                    for i, i_head in enumerate(i_first_batch):
                        

                        ## Flatten across heads, height, and width
                        flattened = i_head.view(-1)

                        
                        ## Calculate statistics
                        i_means.append(torch.nanmean(flattened).item())
                        i_medians.append(torch.nanmedian(flattened).item())

                        # Standard deviation, ignoring NaNs
                        mask = ~torch.isnan(i_head)
                        i_stdevs.append(torch.std(i_head[mask]).item())
                        i_sum_vals.append(torch.sum(i_head[mask]).item())

                        # Max, temporarily replacing NaNs with -inf for calculation
                        i_max_values.append(torch.max(torch.where(torch.isnan(i_head), torch.tensor(float('-inf')), i_head)).item())

                        # Denominator computation for i_head
                        exp_flattened = torch.exp(i_head[mask])
                        sum = torch.sum(exp_flattened)
                        denominator.append(sum.item())

                        # Append statistic to the input list of each head in each layer
                        self.stats['mean'][layer][i].append(torch.nanmean(flattened).item())
                        self.stats['median'][layer][i].append(torch.nanmedian(flattened).item())
                        self.stats['stdev'][layer][i].append(torch.std(i_head[mask]).item())
                        self.stats['max'][layer][i].append(torch.max(torch.where(torch.isnan(i_head), torch.tensor(float('-inf')), i_head)).item())



                    outputs_location = f"transformer.h[{layer}].attn.softmax_layer.outputs"
                    softmax_output = eval(f"self.model.{outputs_location}").to('cpu').to(torch.float32)
                   
                    o_first_batch = softmax_output[0]
                    o_first_batch[o_first_batch == float('-inf')] = float('NaN')
                    for i, o_head in enumerate(o_first_batch):

                        # Step 3: Flatten across heads, height, and width
                        flattened = o_head.view(-1)

                        # Step 4: Calculate statistics
                        ## Calculate statistics
                        o_means.append(torch.nanmean(flattened).item())
                        o_medians.append(torch.nanmedian(flattened).item())
                        # Standard deviation, ignoring NaNs
                        mask = ~torch.isnan(o_head)
                        o_stdevs.append(torch.std(o_head[mask]).item())
                        o_sum_vals.append(torch.sum(o_head[mask]).item())
                        # Max, temporarily replacing NaNs with -inf for calculation
                        o_max_values.append(torch.max(torch.where(torch.isnan(o_head), torch.tensor(float('-inf')), o_head)).item())

                        # Append statistic to the output list of each head in each layer
                        self.stats['o_mean'][layer][i].append(torch.nanmean(flattened).item())
                        self.stats['o_median'][layer][i].append(torch.nanmedian(flattened).item())
                        self.stats['o_stdev'][layer][i].append(torch.std(o_head[mask]).item())
                        self.stats['o_max'][layer][i].append(torch.max(torch.where(torch.isnan(o_head), torch.tensor(float('-inf')), o_head)).item())

                    #BETA GAMMA
                    gamma_location = f"transformer.h[{layer}].attn.softmax_layer.gamma"
                    beta_location = f"transformer.h[{layer}].attn.softmax_layer.beta"

                    gamma = eval(f"self.model.{gamma_location}")
                    gammas.append(gamma[0].item()) # are there more than just gamma 0?
                    # print("gammas",gamma) # are there more than just gamma 0?

                    beta = eval(f"self.model.{beta_location}")
                    betas.append(beta[0].item()) # are there more than just beta 0?
                    # print("betas",beta,) # are there more than just beta 0?

                    self.log_gamma_beta(gamma, beta, self.iter_num, layer)


                self.write_to_csv(self.iter_num,
                                  *i_sum_vals,
                                  *i_means,
                                  *i_medians,
                                  *i_stdevs,
                                  *i_max_values,
                                  *denominator,
                                  prefix="inputs")
                self.write_to_csv(self.iter_num,
                                  *o_sum_vals,
                                  *o_means,
                                  *o_medians,
                                  *o_stdevs,
                                  *o_max_values,
                                  prefix="outputs")
                self.write_to_csv(self.iter_num, *betas, *gammas, prefix="beta_gamma")

            """
            if self.iter_num % 50 == 0:
                inputs = []
                outputs = []

                for layer in range (self.args.n_layer):
                    inputs_location = f"transformer.h[{layer}].attn.softmax_layer.inputs"
                    outputs_location = f"transformer.h[{layer}].attn.softmax_layer.outputs"

                    gamma = eval(f"self.model.{gamma_location}")
                    gammas.append(gamma[0].item()) # are there more than just gamma 0?
                    # print("gammas",gamma) # are there more than just gamma 0?

                    beta = eval(f"self.model.{beta_location}")
                    betas.append(beta[0].item()) # are there more than just beta 0?
                    # print("betas",beta,) # are there more than just beta 0?

                    self.log_gamma_beta(gamma, beta, self.iter_num, layer)


                self.write_to_csv(self.iter_num, *betas, *gammas, prefix="beta_gamma")


            """
            self.iter_num += 1
            local_iter_num += 1

            if self.iter_num > self.args.max_iters:
                self.plot_statistics()
                if self.args.only_save_checkpoint_at_end:
                    checkpoint = {
                        'model': self.raw_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'model_args': self.model_args,
                        'iter_num': self.iter_num,
                        'best_val_loss': self.best_val_loss,
                        'nan_iter_num' : None,
                        'nan' : None,
                        'config': vars(self.args),
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

