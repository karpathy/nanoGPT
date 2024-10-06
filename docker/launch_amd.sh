#!/usr/bin/env bash

CMD=$1
IT_FLAG=${1:+-t}

sudo docker run --privileged --network=host --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 192G \
    --rm ${IT_FLAG:=-it} \
    -v .:/workspace/llm-train-bench/ \
    llm-train-bench $CMD