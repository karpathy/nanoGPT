from dataclasses import dataclass

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

    use_parallel_mlp: bool = False

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

    # Positional Embeddings Variations
    use_abs_pos_embeddings: bool = True # Note: one can use this AND rotary embeddings
    use_fire_embeddings: bool = False
    shared_fire_embeddings: bool = False
    use_rotary_embeddings: bool = False
    rope_variant: str = "rope" # options: "shortrope", "rope"
    shortrope_length: int = 8 # number of embeddings to use in shortrope

    # Structuring Options, remember to compile the model
    use_post_ln: bool = True

    # Layernorm Alternatives and Options
    norm_variant_attn: str = "rmsnorm"
    norm_variant_output: str = "rmsnorm"
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    prmsnorm_pct: float = 0.0625
    krmsnorm_num: float = 10

    # Activation Alternatives
    activation_variant: str = "gelu"

    # Linear Alternatives
    linear_variant: str = "linear"
