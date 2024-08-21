import argparse
from contextlib import nullcontext
import csv
import json
import math
import os
import pickle
import shutil
import sys
import time

from model_info_util.model_info import print_summary, print_module_structure, print_model_blocks, print_model_tree
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from rich import print
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from statistics_util.statistic_plots import initialize_statistics, plot_statistics, create_statistics

from model import GPT, GPTConfig

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
    training_group.add_argument('--eval_only', default=False, action=argparse.BooleanOptionalAction)

    # Checkpoint args
    training_group.add_argument('--only_save_checkpoint_at_end', default=False, action=argparse.BooleanOptionalAction)
    training_group.add_argument('--always_save_checkpoint', default=False, action=argparse.BooleanOptionalAction)
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
    model_group.add_argument('--window_size', default=None, type=int, help="Sliding window size, note this cannot be greater than block size")
    model_group.add_argument('--gate', default=False, action=argparse.BooleanOptionalAction, help="option for gated attention see https://arxiv.org/abs/2306.12929")
    model_group.add_argument('--use_moe', default=False,  action=argparse.BooleanOptionalAction, help="option for Mixture of Experts (MoE) architecture")
    model_group.add_argument('--moe_layer_freq', default=2, type=int, help="set frequency for replacing FFNs with MoE layers")
    model_group.add_argument('--n_experts', default=8, type=int, help="set number of experts per MoE layer")
    model_group.add_argument('--moe_top_k', default=2, type=int)
    model_group.add_argument('--moe_router_scheme', default="softmax", type=str, help="option to set routing scheme for MoE layer, defaults to softmax")

    ## MLP Options
    model_group.add_argument('--use_parallel_mlp', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--mlp_variant", type=str, default="mlp", choices=["mlp", "kan", "swiglu"], help="MLP variation type")

    ## KAN Options
    model_group.add_argument("--kan_poly_order", type=int, default=3, help="Order of KAN non-linearity")
    model_group.add_argument("--kan_base_activation", type=str, default="silu", help="initial KAN activation")
    model_group.add_argument("--kan_middle_layers", type=int, nargs='+', help="List of integers", default=[])

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
    model_group.add_argument("--krmsnorm_quantize_type", type=str, default="none", choices=["int8", "int16", "none"])
    model_group.add_argument('--krmsnorm_enable_gain', default=True, action=argparse.BooleanOptionalAction, help="include gain in kRMSNorm")
    model_group.add_argument("--krmsnorm_selection_type", type=str, default="last", choices=["first", "last", "random"])
    model_group.add_argument("--krmsnorm_recompute_percentage", type=float, default=None, help="percentage needed within the total RMS to not trigger recompute")

    activation_variations = [
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
        ]

    # ACTIVATION VARIATIONS
    model_group.add_argument( "--activation_variant", type=str, default="gelu", choices=activation_variations,)

    # LINEAR VARIATIONS
    linear_variants = ["linear", "bitlinear", "bitlinear_1p58", "bitlinear_optimized", "kan","quantized_linear"]
    model_group.add_argument("--linear_variant_attn", type=str, default="linear", choices=linear_variants)
    model_group.add_argument("--linear_variant_q", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_attn_q in attention (takes precedence over linear_variant_attn)")
    model_group.add_argument("--linear_variant_k", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_attn_k in attention (takes precedence over linear_variant_attn)")
    model_group.add_argument("--linear_variant_v", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_attn_v in attention (takes precedence over linear_variant_attn)")
    model_group.add_argument("--linear_variant_attn_proj", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_proj in attention (takes precedence over linear_variant_attn)")
    model_group.add_argument("--linear_variant_mlp", type=str, default="linear", choices=linear_variants)
    model_group.add_argument("--linear_variant_mlp_up", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_fc in mlp (takes precedence over linear_variant_mlp)")
    model_group.add_argument("--linear_variant_mlp_down", type=str, default=None, choices=linear_variants, help="sets the linear variant for c_proj in mlp (takes precedence over linear_variant_mlp)")
    ## Linear Weight Initialization Options
    model_group.add_argument( "--linear_mean_init", type=float, default=0.0)
    model_group.add_argument( "--linear_std_init", type=float, default=0.02)

    # Quatization

    ## Quantization Method Options
    quant_methods = ["symmetric_quant", "affine_quant", "stochastic_quant"]

    ## WTE
    model_group.add_argument("--quantize_wte", default=None, action=argparse.BooleanOptionalAction, help="Whether the word embedding is quantized")
    model_group.add_argument("--quantize_wte_method", type=str, default="affine_quant", choices=quant_methods, help="function used for word embedding quantization")
    model_group.add_argument("--quantize_wte_bits", type=int, default=8, help="number of bits for word embedding quantization")

    ## WPE
    model_group.add_argument("--quantize_wpe", default=None, action=argparse.BooleanOptionalAction, help="Whether the word position embedding is quantized")
    model_group.add_argument("--quantize_wpe_method", type=str, default="affine_quant", choices=quant_methods, help="function used for position embedding quantization")
    model_group.add_argument("--quantize_wpe_bits", type=int, default=8, help="number of bits for position embedding quantization")

    ## Activations
    model_group.add_argument("--activations_quant_method", type=str, default="affine_quant", choices=quant_methods, help="function used for quantization of activations")

    ### Attention Activations
    model_group.add_argument("--quantize_attn_act", action=argparse.BooleanOptionalAction, default=False, help="quantize all input/output activations in attn")

    #### Whether to do Attention Activation quantization at the Arrow
    model_group.add_argument("--quantize_attn_act_input", action=argparse.BooleanOptionalAction, default=False, help="quantize input activation to attention")
    model_group.add_argument("--quantize_attn_act_qk_mult_q_input", action=argparse.BooleanOptionalAction, default=False, help="quantize query input activation to qk mult")
    model_group.add_argument("--quantize_attn_act_qk_mult_k_input", action=argparse.BooleanOptionalAction, default=False, help="quantize key input activation to qk mult")
    model_group.add_argument("--quantize_attn_act_softmax_input", action=argparse.BooleanOptionalAction, default=False, help="quantize input activation to softmax")
    model_group.add_argument("--quantize_attn_act_pv_mult_p_input", action=argparse.BooleanOptionalAction, default=False, help="quantize softmax input activation to pv mult")
    model_group.add_argument("--quantize_attn_act_pv_mult_v_input", action=argparse.BooleanOptionalAction, default=False, help="quantize value input activation to pv mult")
    model_group.add_argument("--quantize_attn_act_pv_mult_output", action=argparse.BooleanOptionalAction, default=False, help="quantize output activation of pv_mult")
    model_group.add_argument("--quantize_attn_act_output", action=argparse.BooleanOptionalAction, default=False, help="quantize output activation of attention")

    ### Default Precisions for Attention Activations
    model_group.add_argument("--quantize_attn_act_bits", type=int, default=8, help="number of bits for attn quantization")

    ### Overrides for granular Attention Activatinos
    model_group.add_argument("--quantize_attn_act_input_bits", type=int, default=None, help="number of bits for attention input quantization")
    model_group.add_argument("--quantize_attn_act_qk_mult_q_input_bits", type=int, default=None, help="number of bits for qk mult query input quantization")
    model_group.add_argument("--quantize_attn_act_qk_mult_k_input_bits", type=int, default=None, help="number of bits for qk mult key input quantization")
    model_group.add_argument("--quantize_attn_act_softmax_input_bits", type=int, default=None, help="number of bits for softmax input quantization")
    model_group.add_argument("--quantize_attn_act_pv_mult_p_input_bits", type=int, default=None, help="number of bits for pv mult softmax input quantization")
    model_group.add_argument("--quantize_attn_act_pv_mult_v_input_bits", type=int, default=None, help="number of bits for pv mult value input quantization")
    model_group.add_argument("--quantize_attn_act_pv_mult_output_bits", type=int, default=None, help="number of bits for pv mult output quantization")
    model_group.add_argument("--quantize_attn_act_output_bits", type=int, default=None, help="number of bits for attention output quantization")

    ### Whether to use MLP Activations
    model_group.add_argument("--quantize_mlp_act", action=argparse.BooleanOptionalAction, default=False, help="quantize all input/output activations in mlp")
    model_group.add_argument("--quantize_mlp_act_input", action=argparse.BooleanOptionalAction, default=False, help="quantize input activation to mlp")
    model_group.add_argument("--quantize_mlp_act_activation_input", action=argparse.BooleanOptionalAction, default=False, help="quantize input activation to activation function")
    model_group.add_argument("--quantize_mlp_act_activation_output", action=argparse.BooleanOptionalAction, default=False, help="quantize output activation of activation function")
    model_group.add_argument("--quantize_mlp_act_output", action=argparse.BooleanOptionalAction, default=False, help="quantize output activation of mlp")

    ### Default Precisions for MLP Activations
    model_group.add_argument("--quantize_mlp_act_bits", type=int, default=8, help="number of bits for mlp quantization")

    ### Overrides for granular MLP Activatinos
    model_group.add_argument("--quantize_mlp_act_input_bits", type=int, default=None, help="number of bits for mlp input quantization")
    model_group.add_argument("--quantize_mlp_act_activation_input_bits", type=int, default=None, help="number of bits for activation function input quantization")
    model_group.add_argument("--quantize_mlp_act_activation_output_bits", type=int, default=None, help="number of bits for activation function output quantization")
    model_group.add_argument("--quantize_mlp_act_output_bits", type=int, default=None, help="number of bits for mlp output quantization")

    ### Whether activations should be saved
    model_group.add_argument("--store_activations", action=argparse.BooleanOptionalAction, default=False, help="whether the activations should be saved as a buffer and updated through training")

    ## Linear Attn Weight Quantization Precision and Method

    ### Default methods and precisions
    model_group.add_argument("--quantize_linear_method", type=str, default="affine_quant", choices=quant_methods, help="function used for linear quantization")
    model_group.add_argument("--quantize_linear_bits", type=int, default=8, help="number of bits for linear quantization")

    #### Overrides for granular Methods and Precisions
    model_group.add_argument("--quantize_linear_attn_q_method", type=str, default=None, choices=quant_methods, help="function used for c_attn_q quantization")
    model_group.add_argument("--quantize_linear_attn_q_bits", type=int, default=None, help="number of bits for c_attn_q quantization")

    model_group.add_argument("--quantize_linear_attn_k_method", type=str, default=None, choices=quant_methods, help="function used for c_attn_k quantization")
    model_group.add_argument("--quantize_linear_attn_k_bits", type=int, default=None, help="number of bits for c_attn_k quantization")

    model_group.add_argument("--quantize_linear_attn_v_method", type=str, default=None, choices=quant_methods, help="function used for c_attn_v quantization")
    model_group.add_argument("--quantize_linear_attn_v_bits", type=int, default=None, help="number of bits for c_attn_v quantization")

    model_group.add_argument("--quantize_linear_attn_proj_method", type=str, default=None, choices=quant_methods, help="function used for c_proj in attention quantization")
    model_group.add_argument("--quantize_linear_attn_proj_bits", type=int, default=None, help="number of bits for c_proj in attention quantization")

    #### Overrides for Linear MLP Weight Quantization Precision and Method
    model_group.add_argument("--quantize_linear_mlp_up_method", type=str, default=None, choices=quant_methods, help="function used for mlp_up quantization")
    model_group.add_argument("--quantize_linear_mlp_up_bits", type=int, default=None, help="number of bits for mlp_up quantization")
    model_group.add_argument("--quantize_linear_mlp_down_method", type=str, default=None, choices=quant_methods, help="function used for mlp_down quantization")
    model_group.add_argument("--quantize_linear_mlp_down_bits", type=int, default=None, help="number of bits for mlp_down quantization")

    ## Quantized Linear Warmup Iterations -- how many to first use regular linear, before switching to quantized
    model_group.add_argument("--quantization_warmup_iters", type=int, default=100)

    # POSITIONAL EMBEDDING VARIATIONS
    model_group.add_argument('--use_rotary_embeddings', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--sym_rot_num_angles', type=int, default=512, help="number of angles to use for symmetric rope variant")
    model_group.add_argument("--rope_variant", type=str, default="rope", choices=["rope", "soap"])
    model_group.add_argument("--rope_length", type=int, default=None, help="Defaults to all embeddings (if set to None), else must be even.")
    model_group.add_argument('--use_abs_pos_embeddings', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--use_fire_embeddings', default=False, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--shared_fire_embeddings', default=False, action=argparse.BooleanOptionalAction)

    ## Positional Embedding Weight Initialization Options
    model_group.add_argument( "--embedding_mean_init", type=float, default=0.0)
    model_group.add_argument( "--embedding_std_init", type=float, default=0.02)

    # SOFTMAX VARIATIONS

    softmax_variations = [
        "saturatingconsmax",
        "consmax",
        "consmax_v2",
        "consmax_quan",
        "polymax",
        "relumax",
        "vpolymax",
        "exppolymax",
        "strongermax",
        "softermax",
        "sigsoftmax",
        "softmax",
        "softplus",
        "squareplus",
        "exppolymax",
        ]

    ## Selection of softmax variation for attention and output layers
    model_group.add_argument("--softmax_variant_attn", type=str, default="softmax", choices=softmax_variations)
    model_group.add_argument("--softmax_variant_output", type=str, default="softmax", choices=softmax_variations)

    ## Custom Softmax Variation Options
    ### ConSmax and SaturatingConSmax Options
    model_group.add_argument("--consmax_initial_beta", type=float, default=2.5)
    model_group.add_argument("--consmax_initial_gamma", type=float, default=100.0)
    model_group.add_argument('--consmax_use_euler_base', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--consmax_base", type=float, default=2.0)

    ### Special Options for ConSmaxV2
    model_group.add_argument("--consmax_per_head", default=True, action=argparse.BooleanOptionalAction)

    ### Special Options for SaturatingConSmax
    model_group.add_argument("--consmax_saturation", type=float, default=11.0, help="point where we transition from consmax to linear saturatingconsmax, defaults to 11 to approximate e^x sat for fp16")
    model_group.add_argument('--consmax_learnable_beta', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--consmax_learnable_gamma', default=True, action=argparse.BooleanOptionalAction)

    ### Polymax Options
    model_group.add_argument("--polymax_x_intercept", type=float, default=-100.0)
    model_group.add_argument("--polymax_y_intercept", type=float, default=1.0)
    model_group.add_argument("--polymax_power", type=float, default=2.0)
    model_group.add_argument("--polymax_divisor", type=float, default=1000.0)

    ### ReLUMax Options
    model_group.add_argument("--relumax_divisor", type=float, default=256.0)

    ### SigSoftmax Options
    model_group.add_argument('--sigsoftmax_use_euler_base', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--sigsoftmax_base", type=float, default=2.0)

    ### Strongermax Options - Testing Incremental Adjustments to Regular Softmax
    model_group.add_argument("--strongermax_strength", type=float, default=4.0)
    model_group.add_argument('--strongermax_sum_to_1', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--strongermax_divisor", type=float, default=1.0)
    model_group.add_argument('--strongermax_use_xmax', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument('--strongermax_xmax_guess', type=float, default=None)
    model_group.add_argument('--strongermax_overflow_recompute', default=True, action=argparse.BooleanOptionalAction)

    ### ExpPolymax Options
    model_group.add_argument('--exppolymax_use_euler_base', default=True, action=argparse.BooleanOptionalAction)
    model_group.add_argument("--exppolymax_base", type=float, default="4")
    model_group.add_argument("--exppolymax_y_intercept", type=float, default=1.0)
    model_group.add_argument("--exppolymax_power", type=float, default=2.0)
    model_group.add_argument("--exppolymax_divisor", type=float, default=1000.0)

    ### Softermax Specific Options
    model_group.add_argument('--softermax_use_xmax', default=True, action=argparse.BooleanOptionalAction)

    ### SoftPlus Options
    model_group.add_argument('--softplus_divisor', type=float,default=100.0)
    ### SquarePlus Options
    model_group.add_argument('--squareplus_divisor', type=float,default=100.0)

    ### Sequence Length Division https://arxiv.org/abs/2309.
    model_group.add_argument('--div_by_seq_len', default=False, action=argparse.BooleanOptionalAction)

    # Gradient Checkpointing
    training_group.add_argument('--use_gradient_checkpointing', default=False, action=argparse.BooleanOptionalAction, help="Memory efficient training, but takes longer time to train due to trading compute time for memory efficiency. For best memory tradeoff omit the --compile flag. For medium memory tradeoff add --compile.")

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

    # Module And Parameter Logging and Plots of Summary Statistics
    model_group.add_argument('--softmax_io_logging', default=False, action=argparse.BooleanOptionalAction, help="logs inputs and outputs of supported softmaxes")
    model_group.add_argument('--consmax_beta_gamma_logging', default=False, action=argparse.BooleanOptionalAction, help="logs beta and gamma")
    logging_group.add_argument('--create_statistics', default=False, action=argparse.BooleanOptionalAction)
    logging_group.add_argument('--plot_statistics', default=False, action=argparse.BooleanOptionalAction)

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

    ### Create model from json config file & save config file to json
    logging_group.add_argument('--load_config_json', type=str, help="Option to load model parameters from existing json file")
    logging_group.add_argument('--save_config_json', type=str, help="Option to save model parameters as new config json file")

    # Visualization args
    logging_group.add_argument('--statistic', choices=[ 'input_mean', 'input_median', 'input_stdev', 'input_max', 'input_min', 'output_mean', 'output_median', 'output_stdev', 'output_max', 'output_min', 'all_stats', 'input_all','output_all' ], default='input_mean', help='Select one or all statistics to display, e.g., --statistic input_min, or --statistic all_stats')
    logging_group.add_argument('--graph_type', choices=[ "heatmap", "plot", "boxplot", "all" ], default='no_graph', help='Select one of the graph types to display, e.g., --graph_type heatmap, or --graph_type plot')
    logging_group.add_argument('--box_plot_interval', default=1000, type=int, help='Instead of using mean/median/stdev statistics, create box plot of all input/output values at certain intervals of iteration')
    logging_group.add_argument('--box_plot_statistic', choices=['input', 'output', 'all'], default='', help='Select input or output statistic to display in boxplot')

    # Model Parameter Distribution
    logging_group.add_argument('--print_model_info', default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.load_config_json is not None:
        with open(args.load_config_json, 'r') as config_file:
            config = json.load(config_file)

        # Update the args namespace with values from the JSON file
        for key, value in config.items():
            setattr(args, key, value)

    # Save all params to provided json if flag is present
    if args.save_config_json is not None:
        with open(args.save_config_json, 'w') as json_file:
            json.dump(vars(args), json_file)

    return args, model_group, training_group, logging_group

class Trainer:

    def __init__(self, args, model_group, training_group, logging_group):
        self.args = args
        self.model_group = model_group
        self.training_group = training_group
        self.logging_group = logging_group

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
            self.args.gradient_accumulation_steps //= self.ddp_world_size
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

        # Model settings
        # TODO only add if they are defined from the argparse
        self.model_args = {action.dest: getattr(self.args, action.dest) for action in self.model_group._group_actions}
        self.model_args['vocab_size'] = None
        self.model_args['use_gradient_checkpointing'] = self.args.use_gradient_checkpointing

        # Training settings
        self.training_args = {action.dest: getattr(self.args, action.dest) for action in self.training_group._group_actions}

        if self.args.init_from == 'scratch':
            self.model_args['vocab_size'] = self.get_vocab_size_from_meta()

            # Save full configuration used for training
            config_json = {**self.model_args, **self.training_args}
            with open(self.args.out_dir + "/full_config.json", "w") as configuration_file:
                json.dump(config_json, configuration_file, indent=4)
            with open(self.args.out_dir + "/best_val_loss_and_iter.txt", 'w') as file:
                print("resetting best val loss file")

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
            ## TODO: Add means here to udpate the resume for: block size (finetune for longer context), rotary type, rope length, softmax type, etc.
            ## TODO: Add ability here to swap WTE factors.
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

        # Print the model summary
        if self.args.print_model_info:
            print_summary(self.model)
            print_model_blocks(self.model)
            print_module_structure(self.model)
            print_model_tree(self.model, print_params=True)

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
        # Save a copy of meta.pkl tokenization into the output folder
        self.copy_file_to_directory(meta_path, self.args.out_dir)
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                if 'vocab_size' in meta:
                    return meta['vocab_size']
                else:
                    sys.exit(f"Error: 'vocab_size' key not found in {meta_path}")
        else:
            sys.exit(f"Error: File not found - {meta_path}")

    def copy_file_to_directory(self, src_file, dest_dir):
        try:
            # Ensure the destination directory exists
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # Copy the file
            shutil.copy(src_file, dest_dir)
            print(f"File {src_file} copied to {dest_dir}")
        except Exception as e:
            print(f"Error copying file: {e}")

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


    def log_gamma_beta(self, gamma, beta, iter_num, layer_num, head_num=None):
        if self.args.tensorboard_log:
            if head_num:
                self.writer.add_scalars(
                        "gammas",
                        {"gamma_L" + str(layer_num) + "_H" + head_num: gamma},
                        iter_num
                        )
                self.writer.add_scalars(
                        "betas",
                        {"beta_L" + str(layer_num) + "_H" + head_num: beta},
                        iter_num
                        )
            else:
                self.writer.add_scalar( "gamma_L" + str(layer_num), gamma, iter_num)
                self.writer.add_scalar( "beta_L" + str(layer_num), beta, iter_num)

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

    def train(self):
        self.X, self.Y = self.get_batch('train')
        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0
        num_steps_with_worse_loss = 0
        graph_y_labels = []
        for layer in range(self.args.n_layer):
            for head in range(self.args.n_head):
                graph_y_labels.append(f"Layer {layer} Head {head}")

        while True:
            lr = self.get_lr(self.iter_num) if self.args.decay_lr else self.args.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if self.iter_num % self.args.eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                self.log_metrics(losses, lr, running_mfu, self.iter_num)

                if math.isnan(losses["val"]):
                    checkpoint = {
                        'model': self.raw_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'model_args': self.model_args,
                        'iter_num': self.iter_num,
                        'best_val_loss': self.best_val_loss,
                        'nan_iter_num' : 0,
                        'nan' : True,
                        'config': vars(self.args),
                    }
                    torch.save(checkpoint, os.path.join(self.args.out_dir, 'ckpt.pt'))
                if losses['val'] < self.best_val_loss or self.args.always_save_checkpoint:
                    if losses['val'] < self.best_val_loss:
                        self.iter_num_best_val_loss = self.iter_num
                        self.best_val_loss = losses['val']
                        # Save best validation loss
                        with open(os.path.join(self.args.out_dir, 'best_val_loss_and_iter.txt'), "w") as best_loss_file:
                            best_loss_file.write(str(self.best_val_loss.item())+","+str(self.iter_num))
                        # Reset early exit counter
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
                        # Save checkpoint
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


            if self.args.create_statistics:
                create_statistics(self, graph_y_labels)


            self.iter_num += 1
            local_iter_num += 1

            # End of training actions
            if self.iter_num > self.args.max_iters:
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

        if self.args.plot_statistics:
            plot_statistics(self.args, self.stats, graph_y_labels)

        if self.args.tensorboard_log:
            self.writer.flush()
            self.writer.close()

        if self.args.wandb_log and self.master_process:
            import wandb
            wandb.log({"finished": True})
            wandb.finish()

def main():
    args, model_group, training_group, logging_group = parse_args()
    trainer = Trainer(args, model_group, training_group, logging_group)
    trainer.train()

    if trainer.ddp:
        destroy_process_group()

    if args.tensorboard_log:
        trainer.writer.flush()
        trainer.writer.close()

if __name__ == '__main__':
    main()

