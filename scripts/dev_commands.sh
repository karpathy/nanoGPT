## TODO Document
function start_ngpt_container() {
        local DEV_REPO_PATH=/mnt/nanoGPT
        local DEV_IMAGE_NAME=nemo_framework_dev_env
        # local DEV_IMAGE_NAME=nvcr.io/nvidian/nemo-architecture-search:v24.03.01
        docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                -p 60122:22 -p 8890:8890 --mount source=/mnt,target=/mnt,type=bind --mount source=${DEV_REPO_PATH},target=/workspaces,type=bind \
                ${DEV_IMAGE_NAME} /usr/sbin/sshd -D
}

function start_aquarium_jupyter_server() {
  PYTHONPATH=/workspaces/nanoGPT
  local DOCKER_ID="$(docker ps -f ancestor=nemo_framework_dev_env -q)"
  docker exec -it $DOCKER_ID /bin/bash -c "cd /workspaces/nanoGPT && PYTHONPATH=${PYTHONPATH} jupyter server --ip 0.0.0.0 --no-browser --allow-root --port 8890"
}