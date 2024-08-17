# Adopted from NanoGPT commit https://github.com/karpathy/nanoGPT/commit/9755682b981a45507f6eb9b11eadef8cb83cebd5

Rocm 6.2
```bash
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 192G -v .:/var/lib/jenkins/nanoGPT rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
cd /var/lib/jenkins/nanoGPT
pip install tiktoken datasets
pip install numpy==1.22.4
```

Pytorch 24.07 is the recommended image for H100
```bash
docker run --gpus all -it --ipc=host --shm-size=192G --rm -v .:/workspace/nanoGPT nvcr.io/nvidia/pytorch:24.07-py3
cd /workspace/nanoGPT
```

## Downloading dataset (inside container)
- note that since yr local_dir is mounted at /workspace/nanoGPT, the dataset will be cached there between runs and between container stop and starts as long as you remount it.
The following command will downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin`
```sh
python data/openwebtext/prepare.py
```

## GPT2 125M Running torch.compile
```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

## Gpt2 125M Run Command Eager
```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_eager.py
```