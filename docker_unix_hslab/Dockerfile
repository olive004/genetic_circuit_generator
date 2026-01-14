# syntax=docker/dockerfile:1

# https://quay.io/repository/biocontainers/intarna
FROM quay.io/biocontainers/intarna:3.4.1--pl5321hdcf5f25_0 AS intarna_build

FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04 AS workdir_build

################################################################################
#  Install python dependencies
################################################################################
WORKDIR /usr/lib
COPY ./requirements.txt /usr/lib/install_requirements/requirements.txt

# Install requirements
RUN apt-get update && \
    yes | apt-get install python3-pip
RUN yes | apt install git
RUN cd /usr/lib/install_requirements && \
    pip install -r ./requirements.txt 
# RUN pip install jax==0.4.13
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install git+https://github.com/Steel-Lab-Oxford/core-bioreaction-simulation.git@f903c39872de43e28b56653efda689bb082cb592#egg=bioreaction
RUN export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/

################################################################################
#  setup environment
################################################################################
ENV ROOT_DIR=/workdir
ENV PYTHONPATH=/workdir
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV MNT=./..
WORKDIR /workdir

################################################################################
#  Copy IntaRNA dependencies
################################################################################
COPY --from=intarna_build /usr/local/bin/IntaRNA /usr/local/bin/
COPY --from=intarna_build /usr/local/include/IntaRNA /usr/local/include/
COPY --from=intarna_build /usr/local/lib/* /usr/local/lib/
