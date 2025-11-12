import time

out_dir = 'out-fairytails'  
eval_interval = 5
eval_iters = 40
wandb_log = False
wandb_project = 'fairytails'  
wandb_run_name = 'ft-' + str(time.time())

dataset = 'fairytails'        
init_from = 'gpt2'    

always_save_checkpoint = False

batch_size = 1
gradient_accumulation_steps = 32
max_iters = 5    
learning_rate = 3e-5
decay_lr = False
