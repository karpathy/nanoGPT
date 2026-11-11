# multichannel EEG: windows of [channels, frames] from data/eeg_multi, masked-token prediction (copied from train_eeg_single.py)

out_dir = 'out/eeg-multi'
eval_interval = 250
eval_iters = 200
log_interval = 1

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'nanogpt-eeg-cw'
wandb_run_name = 'eeg-multi-none-cw'

dataset = '/data/groups/bci/datasets/tmp_dac_tokens_jonas/eeg_multi' # (CW) - was 'eeg_multi'
gradient_accumulation_steps = 1
batch_size = 16             # 16 windows * 3008 tokens ~ 48k tokens per step (single-channel: 64 * 256 = 16k)
block_size = 3008           # max_channels * window_frames = 94 * 32

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

pos_encoding = 'sinusoidal' #jm
use_codebook = True   #jm tokens enter as their DAC codebook vectors (data/<dataset>/codebook.npy)
eeg_dataloader = True   #jm windows from the per-recording grids
bidirectional = True   #jm
mask_ratio = 0.15      #jm

multichannel = True         #jm windows of [channels, frames]
max_channels = 94           #jm at most one channel per standard location
window_frames = 32          #jm 4 s per window (8 frames per second)
channel_select = 'standard94'  #jm one channel per standard location, within 2 cm
channel_encoding = 'none'   #jm the control; then 'learned'
