FROM nvcr.io/nvidia/pytorch:24.09-py3
# FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

RUN pip install uv
RUN uv pip install --system ipython pytest fire pydantic

RUN apt update && apt install -y git
RUN pip install git+https://github.com/pytorch-labs/attention-gym.git

WORKDIR /workspace/llm-train-bench/

CMD ["/usr/bin/bash"]
