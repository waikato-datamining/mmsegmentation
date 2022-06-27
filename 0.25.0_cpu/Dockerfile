ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install MMCV
RUN pip install --no-cache-dir mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Install MMDetection
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation && \
    cd /mmsegmentation && \
    git checkout tags/v0.25.0
WORKDIR /mmsegmentation
RUN pip install --no-cache-dir -v -e . && \
    pip install --no-cache-dir python-image-complete "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis "redis-docker-harness==0.0.1"

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
