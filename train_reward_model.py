
from trainer import RewardModelTrainer
import yaml

# assert enc.decode(enc.encode("hello world")) == "hello world"

with open('config.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
    # nested dictionary structure
    config = {}               
    for k, v in conf.items():
        for k2, v2 in v.items():
            config[k2] = v2
    # convert to dotdict
print(config)
trainer = RewardModelTrainer(config)

trainer.train()

