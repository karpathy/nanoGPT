# derived from https://github.com/h2oai/h2ogpt/blob/main/Dockerfile

# runs in Dockers running on host with Nvidia Container Toolkit

# docker build -t niccolox/nanogpt:0.1 .

# docker run -it --runtime=nvidia --gpus all niccolox/nanogpt:0.1

# nvidia-smi

FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            ca-certificates \
            software-properties-common \
            curl \
            git \
            python3 \
            python3-pip \
            python3-dev \
            vim \
            wget && \
    pip3 install --upgrade wheel setuptools && \
    pip3 install --upgrade pip

COPY . /workspace/
WORKDIR /workspace

#RUN pip install -r requirements.txt
RUN pip install torch numpy transformers datasets tiktoken wandb tqdm

RUN chmod -R a+rwx /workspace

RUN python3 data/shakespeare_char/prepare.py
# RUN python3 train.py config/train_shakespeare_char.py
# RUN python3 sample.py --out_dir=out-shakespeare-char

# RUN python3 data/openwebtext/prepare.py
# RUN torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py