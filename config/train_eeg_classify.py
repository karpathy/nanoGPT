# classifier on labelled DAC-token windows (data/eeg_labelled/prepare.py), from a pretrained eeg-multi checkpoint or from labels only
out_dir = 'out/eeg-classify'
eval_interval = 250
log_interval = 10
always_save_checkpoint = False  # keep the checkpoint with the best val loss

wandb_log = True
wandb_project = 'nanogpt-eeg'
wandb_run_name = 'eeg-classify'

dataset = 'eeg_hubner_p3'       # data/<dataset>/{windows.npz, train, val, meta.pkl, codebook.npy}: a symlink into processed/dac_tokens/
task = 'classify'
pretrained_ckpt = ''            # '' = labels only | out/eeg-multi-rope-d384-l6-r3/ckpt.pt (masked) | out/eeg-multi-ntp-rope-d384-l6-r4/ckpt.pt (next-token)
freeze_backbone = False         # True = linear probe on the pretrained features

gradient_accumulation_steps = 1
batch_size = 64
block_size = 3008               # >= max_channels * frames per window (94 * 8 for 1 s P3 windows; 5 s windows need 94 * 40 = 3760)

n_layer = 6                     # must match pretrained_ckpt
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-4            # fine-tuning: 10x below pretraining
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-5
beta2 = 0.99
warmup_iters = 100

pos_encoding = 'sinusoidal'
use_codebook = True
eeg_dataloader = True
multichannel = True
bidirectional = True            # full attention over the window; task='classify' applies no masking
max_channels = 94
channel_select = 'standard94'
channel_encoding = 'rope4d'
