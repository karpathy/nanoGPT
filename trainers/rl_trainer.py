import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import time, os
from model import RLHF
from trainers.trainer import Trainer

# TODO: this works but is currently crude and incomplete, critic implementation plus PPO are obvious next steps
class PolicyGradientTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.mode = 'RL'
    
    def train(self):

        self.setup_ddp()

        ctx, meta_vocab_size = self.setup()

        # model init
        model = self.init_model()

        model = RLHF(model, self.mode, discrete_reward=self.config['discrete_reward'])

        if self.config['init_multihead_from'] == 'scratch':
            print("initializing multihead from scratch")
        else:
            if self.config['init_multihead_from'] == 'resume':
                print(f"Resuming training from {self.config['out_dir_multihead']}")
                # resume training from a checkpoint.
                ckpt_path = os.path.join(self.config['out_dir_multihead'], 'ckpt.pt')
                checkpoint = torch.load(ckpt_path, map_location=self.device)      
                state_dict = checkpoint['model']
                # fix the keys of the state dictionary :(
                # honestly no idea how checkpoints sometimes get this prefix, have to debug more
                unwanted_prefix = '_orig_mod.'
                for k,v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                model.load_state_dict(state_dict)

        
        if self.config['hard_code_reward']:
            reward_model = None
            print('Using hard-coded reward')
        else:
            print('Using learned reward model')
            if self.config['separate_reward_model']:
                import copy
                reward_model = copy.deepcopy(model)
                print('Reward model instantiated separately')
            else:
                reward_model = model
                print('Reward model and actor model share backbone')
            reward_model.to(self.device)
        
        model.to(self.device)
        
        # actor_optimizer = torch.optim.AdamW(model.model.policy_head.parameters(), lr=1e-2)
        actor_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        last_time = time.time()
        rews_all = []
        max_iters = 100000
        X, Y = self.get_batch('train') # fetch the very first batch
        t0  = time.time()
        for iter in range(max_iters):
            
            states, log_probs, log_probs_reference, rewards, advantages = model.generate(
                X, self.block_size, self.device, self.block_size, reward_model=reward_model, hard_code_reward=self.config['hard_code_reward'])

            # minus KL divergence
            rets = advantages * log_probs.squeeze() #- 1*(log_probs-log_probs_reference) #- 0.05*log_probs
            actor_loss = -rets.sum()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optimizer.step()

            torch.mean(rewards)

            rews_all.append(rewards.mean().detach().cpu().numpy())

            if iter % 1000 == 0:
                t1 = time.time()
                print(f'iter: {iter}, time: {t1-t0}')
                # print(actor_loss, critic_loss)
                print(f'Actor loss: {actor_loss}, iter: {iter}')
                print(f'rets: {np.mean(rews_all[-1000:])}')
                current_time = time.time()
                # print(current_time - last_time)
                last_time = current_time
                text = model.generate(X, self.block_size, self.device, self.block_size, reward_model=reward_model)[0]
                for i in range(1):
                    text_i = text[i,:]
                    # print(reward(text_i))
                    try:
                        print(self.enc.decode(text_i.tolist()))
                    except:
                        continue 


class GumbelTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.mode = 'RL'
    
    def train(self):

        self.setup_ddp()

        ctx, meta_vocab_size = self.setup()

        # model init
        model = self.init_model()

        model = RLHF(model, self.mode, discrete_reward=self.config['discrete_reward'])

        if self.config['init_multihead_from'] == 'scratch':
            print("initializing multihead from scratch")
        else:
            if self.config['init_multihead_from'] == 'resume':
                print(f"Resuming training from {self.config['out_dir_multihead']}")
                # resume training from a checkpoint.
                ckpt_path = os.path.join(self.config['out_dir_multihead'], 'ckpt.pt')
                checkpoint = torch.load(ckpt_path, map_location=self.device)      
                state_dict = checkpoint['model']
                # fix the keys of the state dictionary :(
                # honestly no idea how checkpoints sometimes get this prefix, have to debug more
                unwanted_prefix = '_orig_mod.'
                for k,v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                model.load_state_dict(state_dict)

        separate_reward_model = True     
        if separate_reward_model:
            print('Reward model instantiated as copy')
            import copy
            reward_model = copy.deepcopy(model)
        else:
            reward_model = model
        model.to(self.device)
        reward_model.to(self.device)

        gumbel_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        last_time = time.time()
        rews_all = []
        max_iters = 100000
        X, Y = self.get_batch('train') # fetch the very first batch
        t0  = time.time()
        for iter in range(max_iters):
            
            states, rewards = model.generate_gumbel(X, self.block_size, self.device, self.block_size, reward_model=reward_model)


            loss = -rewards.mean()
            gumbel_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gumbel_optimizer.step()

            torch.mean(rewards)

            rews_all.append(rewards.mean().detach().cpu().numpy())

            if iter % 1000 == 0:
                t1 = time.time()
                print(f'iter: {iter}, time: {t1-t0}')
                print(f'rets: {np.mean(rews_all[-1000:])}')
                current_time = time.time()
                # print(current_time - last_time)
                last_time = current_time
                text = model.generate(X, self.block_size, self.device, self.block_size, reward_model=reward_model)[0]
                for i in range(1):
                    text_i = text[i,:]
                    # print(reward(text_i))
                    try:
                        print(self.enc.decode(text_i.tolist()))
                    except:
                        continue 