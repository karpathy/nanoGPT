# SFT 配置覆盖：GPT-2 124M + Stanford Alpaca
# 用法：python train_sft.py config/finetune_alpaca.py
# 也可继续在命令行追加覆盖，例如：
#   python train_sft.py config/finetune_alpaca.py --max_iters=2000 --learning_rate=5e-5

out_dir = 'out-alpaca'
eval_interval = 50
eval_iters = 20
always_save_checkpoint = False

dataset = 'alpaca'
init_from = 'gpt2'  # GPT-2 124M

# batch / 梯度累积：4 * 8 = 32 等效 batch
batch_size = 4
gradient_accumulation_steps = 8
block_size = 512
max_iters = 1000  # 约 1 epoch（train.bin ~13M tokens, 32*512=16384 tokens/iter → ~800 iters/epoch）

# 小学习率 + cosine decay（SFT 标配，防灾难性遗忘）
learning_rate = 3e-5
decay_lr = True
warmup_iters = 50
lr_decay_iters = 1000
min_lr = 3e-6

weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

dropout = 0.1  # 小数据 SFT 开 dropout 防过拟合
bias = False

device = 'cuda'
compile = False
