from dataclasses import dataclass, field
from typing import List

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_kv_group: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    window_size: int = 128
    gate: bool = False

    # Training options
    ## Gradient Checkpointing - More memory efficient (can do long contexts), but is slower
    use_gradient_checkpointing: bool = False

    # MLP Options
    use_parallel_mlp: bool = False
    mlp_variant: str = "mlp"

    ## KAN Option
    kan_poly_order: int = 3
    kan_base_activation: str = "silu"
    kan_middle_layers: List[int] = field(default_factory=lambda: [])

    # Shared parameters
    # MLP
    shared_mlp_size: int = 1
    shared_mlp_sym: bool = False
    # ATTN
    shared_attn_size: int = 1
    shared_attn_sym: bool = False

    # Softmax Alternatives and Options
    softmax_variant_attn: str = "softmax" # Choices: "softmax" "softermax" "sigsoftmax" "polymax" "strongermax" "consmax"
    softmax_variant_output: str = "softmax" # Choices: "softmax" "softermax" "sigsoftmax" "polymax" "strongermax" "consmax"

    ## General Options
    div_by_seq_len: bool = False # for supported functions will divide by seq length

    ## ConSmax Options
    consmax_initial_beta: float = 2.0 # beta adjustment
    consmax_initial_gamma: float = 100.0 # denominator adjustment
    consmax_use_euler_base: bool = True # use 'e' as base for ConSmax, default
    consmax_base: float = 2.0 # base to utilize for ConSmax

    ## SaturatingConSmax special options (otherwise same as ConSmax)
    consmax_saturation: float = 11.0 # for SaturatingConSmax saturation point
    consmax_learnable_beta: bool = True
    consmax_learnable_gamma: bool = True

    ## Softermax options
    softermax_use_xmax: bool = True # Softermax Option active is softermax selected - True: uses (x - x_max) normalization; False: removes normalization (potential overflow)

    ## Polymax options
    polymax_x_intercept: float = -100.0
    polymax_y_intercept: float = 1.0
    polymax_power: float = 2.0
    polymax_divisor: float = 1000.0

    ## SigSoftmaxBase
    sigsoftmax_use_euler_base: bool = True # use 'e' as base for Constantmax
    sigsoftmax_base: float = 2.0 # denominator to utilize for Constantmax

    ## Strongermax options
    strongermax_strength: float = 2.0 # Softermax with option of 'stronger' (larger integer) bases
    strongermax_sum_to_1: bool = False # Softermax with option of 'stronger' (larger integer) bases
    strongermax_divisor: float = 1.0 # Softermax with option of 'stronger' (larger integer) bases
    strongermax_use_xmax: bool = True # Softermax with option of 'stronger' (larger integer) bases

    ## ExpPolymax options
    exppolymax_use_euler_base: bool = True
    exppolymax_base: float = 2.719
    exppolymax_y_intercept: float = 1.0
    exppolymax_power: float = 2.0
    exppolymax_divisor: float = 1.0

    ## Softplus options
    softplus_divisor: float = 100.0

    ## Squareplus options
    squareplus_divisor: float = 100.0

    # Positional Embeddings Variations
    use_abs_pos_embeddings: bool = True # Note: one can use this AND rotary embeddings
    use_fire_embeddings: bool = False
    shared_fire_embeddings: bool = False
    use_rotary_embeddings: bool = False
    sym_rot_num_angles: int = 512
    rope_variant: str = "rope" # options: "shortrope", "rope"
    shortrope_length: int = 8 # number of embeddings to use in shortrope

    ## Embedding Intialization Options
    embedding_mean_init: float= 0.0
    embedding_std_init: float= 0.02

    # Structuring Options, remember to compile the model
    use_post_ln: bool = True

    # Layernorm Alternatives and Options
    norm_variant_attn: str = "rmsnorm"
    norm_variant_output: str = "rmsnorm"
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    prmsnorm_pct: float = 0.0625
    krmsnorm_num: float = 10
    krmsnorm_quantize_type: str = 'int8'
    krmsnorm_enable_gain: bool = True
    krmsnorm_selection_type: str = 'last'
    krmsnorm_recompute_percentage: float = 0.05

    # Activation Alternatives
    activation_variant: str = "gelu"

    # Linear Alternatives
    linear_variant: str = "linear"

    ## Linear Initialization Options
    linear_mean_init: float= 0.0
    linear_std_init: float= 0.02

