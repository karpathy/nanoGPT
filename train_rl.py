import numpy as np
import tiktoken
import time
import torch

from model import GPTConfig, GPT
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from trainer import RLTrainer

# load config.yaml from current directory
with open('config_rl.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
    # nested dictionary structure
    config = {}               
    for k, v in conf.items():
        for k2, v2 in v.items():
            config[k2] = v2
    # convert to dotdict
print(config)
trainer = RLTrainer(config)

trainer.train()




# enc = tiktoken.get_encoding("gpt2")


# device2 = 'cuda:1'
# a = ActorModel(gptconf)
# a.to(device2)
# actor_optimizer = torch.optim.AdamW(a.lm_head.parameters(), lr=1e-2)

# last_time = time.time()
# rets_all = []
# max_iters = 100000
# for iter in range(max_iters):

#   states, probs = a.generate(xb.to(device2), block_size, device2)

#   rewards = model(torch.tensor(states))[0][:,1].unsqueeze(-1)

#   rets = rewards * probs.squeeze()*1000 #- 0.05*log_probs
#   actor_loss = -rets.sum()
#   actor_optimizer.zero_grad(set_to_none=True)
#   actor_loss.backward()
#   actor_optimizer.step()

#   rets_all.append(rewards.mean().detach().cpu().numpy())

#   if iter % 1000 == 0:
#     # print(actor_loss, critic_loss)
#     print(f'Actor loss: {actor_loss}, iter: {iter}')
#     print(f'rets: {np.mean(rets_all[-1000:])}')
#     current_time = time.time()
#     # print(current_time - last_time)
#     last_time = current_time
#     text = a.generate(xb, block_size, device2)[0]
#     for i in range(1):
#       text_i = text[i,:]
#       # print(reward(text_i))
#       print(enc.decode(text_i.tolist()))
