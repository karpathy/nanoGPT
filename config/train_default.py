import ml_collections as mlc

def get_config():
    config = mlc.ConfigDict()
    config.out_dir = 'out'
    config.eval_interval = 2000
    config.log_interval = 1
    config.eval_iters = 200
    config.eval_only = False # if True, script exits right after the first eval
    config.always_save_checkpoint = True # if True, always save a checkpoint after each eval
    
    # wandb logging
    config.wandb_log = False # disabled by default
    config.wandb_entity = None
    config.wandb_project = 'owt'
    config.wandb_run_name = 'gpt2' # 'run' + str(time.time())
    
    # data
    config.dataset = 'openwebtext'
    config.batch_size = 12
    config.block_size = 1024
    
    # model
    config.device = 'cuda:0'
    config.init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    config.dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    config.n_layer = 12
    config.n_head = 12
    config.n_embd = 768
    
    # adamw optimizer
    config.learning_rate = 6e-4 # max learning rate
    config.max_iters = 400000 # total number of training iterations
    config.weight_decay = 1e-2
    config.betas = (0.9, 0.95)
    
    # learning rate decay settings
    config.decay_lr = True # whether to decay the learning rate
    config.warmup_iters = 2000 # how many steps to warm up for
    config.lr_decay_iters = 400000 # should be ~= max_iters per Chinchilla
    config.min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    
    # DDP settings
    config.backend = 'nccl' # 'nccl', 'gloo', etc.
    config.compile = True # use PyTorch 2.0 to compile the model to be faster
    return config