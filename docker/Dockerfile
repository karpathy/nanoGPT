FROM nvcr.io/nvidia/pytorch:24.09-py3

RUN pip install uv
RUN uv pip install --system ipython pytest fire pydantic
RUN uv pip install --system git+https://github.com/pytorch-labs/attention-gym.git

WORKDIR /workspace/llm-train-bench/

CMD ["/usr/bin/bash"]
