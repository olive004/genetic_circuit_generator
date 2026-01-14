# A pre-req for using gpus here is the NVIDIA Docker Container Toolkit

sudo docker pull docker/dockerfile:1
sudo docker pull quay.io/biocontainers/intarna:3.4.1--pl5321hdcf5f25_0

# If image not built yet
if [ "$(id -un)" != "wadh6511" ]; then
    sudo docker pull nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04
    source_directory="docker_unix_hslab"
else
    sudo docker pull nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
    source_directory="docker_unix"
fi

sudo docker build -t genetic_glitch:latest $source_directory

cp ./requirements.txt ./$source_directory

sudo docker create -it \
--rm \
--gpus all \
--name gcg2 \
--mount type=bind,source="$(pwd)",target=/workdir \
genetic_glitch:latest
sudo docker container start gcg2
sudo docker exec -it gcg2 bash setup/post_install.sh
sudo docker exec -it gcg2 /bin/bash
# sudo docker container stop gcg2

if [ "$(id -un)" != "wadh6511" ]; then
    bash $source_directory/post_install2.sh
fi
