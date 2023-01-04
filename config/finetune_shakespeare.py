import time

from config.train_default import get_config as default_config

def get_config():
    config = default_config()
    

    config.out_dir = 'out-shakespeare'
    config.eval_interval = 200
    config.wandb_log = True # feel free to turn on
    config.wandb_project = 'shakespeare'
    config.wandb_run_name = 'ft-' + str(time.time())
    config.compile = False # takes too little time to finetune, not worth it

    # save a nice and overfit checkpoint that
    # will only speak Shakespeare and forgets
    # everything else about the world #dark
    config.always_save_checkpoint = True

    config.dataset = 'shakespeare'
    config.init_from = 'gpt2-xl'
    config.batch_size = 1
    config.block_size = 512

    config.learning_rate = 1e-5
    config.max_iters = 1000
    config.decay_lr = False
    
    return config