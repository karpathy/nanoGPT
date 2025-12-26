DEV.md

docker build -t niccolox/nanogpt:0.1 .

docker run -it --runtime=nvidia --gpus all niccolox/nanogpt:0.1

nvidia-smi

python3 data/shakespeare_char/prepare.py
python3 train.py config/train_shakespeare_char.py
python3 sample.py --out_dir=out-shakespeare-char

python3 data/openwebtext/prepare.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py