import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from trainers.rl_trainer import PolicyGradientTrainer, GumbelTrainer

# load config.yaml from current directory
with open('config/config_rl.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
    # nested dictionary structure
    config = {}               
    for k, v in conf.items():
        for k2, v2 in v.items():
            config[k2] = v2
    # convert to dotdict

if config['method'] == 'gumbel':
    print('Using Gumbel method')
    assert config['hard_code_reward'] == False, 'hard_code_reward must be False for Gumbel method'
    trainer = GumbelTrainer(config)
elif config['method'] == 'pg':
    print('Using Policy Gradient method')
    trainer = PolicyGradientTrainer(config)
else:
    raise NotImplementedError
        
trainer.train()