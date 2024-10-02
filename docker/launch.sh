#!/usr/bin/env bash

CMD=$1
IT_FLAG=${1:+-t}

docker run --rm ${IT_FLAG:=-it} --gpus all --ipc=host --shm-size=192G \
    -v .:/workspace/llm-train-bench/ \
    kimbochen/llm-train-bench $CMD
