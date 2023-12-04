#!/bin/bash

docker run --rm -ti -u $(id -u):$(id -g) \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /etc/passwd:/etc/passwd:ro \
    -v $(pwd):/workspace \
    -e HOME=/workspace \
    trainer
