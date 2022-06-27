ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:11.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.11.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV FORCE_CUDA="1"

# Install MMCV
RUN pip install mmcv-full==1.4.4

# Install MMDetection
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation && \
    cd /mmsegmentation && \
    git checkout tags/v0.25.0
WORKDIR /mmsegmentation
RUN pip install -r requirements/build.txt && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir python-image-complete "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis "redis-docker-harness==0.0.1"

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
