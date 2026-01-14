@REM # See README.md in the root project directory for docker setup instructions
docker pull docker/dockerfile:1
docker pull quay.io/biocontainers/intarna:3.4.1--pl5321hdcf5f25_0
docker pull nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

@REM # If image not built yet
docker build -t genetic_glitch:latest .\docker_windows

@REM # cp ./requirements.txt ./docker_windows

docker create -it ^
--rm ^
--gpus all ^
--name gcg ^
--mount=type=bind,source=%cd%,target=/workdir ^
genetic_glitch:latest
docker container start gcg
docker exec -it gcg /bin/bash 
@REM # docker container stop gcg


@REM # Docusaurus
@REM # 1. Started by installing Node.js and following installation guide https://github.com/nodejs/help/wiki/Installation
@REM # 1.a Added to end of ~/.profile 
@REM # 1.b Refresh . ~/.profile and test installation with npx -v
