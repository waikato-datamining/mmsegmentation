ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2004
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV FORCE_CUDA="1"

# Install MMCV
RUN pip install --no-cache-dir mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.11.0/index.html

# Install MMDetection
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation && \
    cd /mmsegmentation && \
    git checkout tags/v0.30.0
WORKDIR /mmsegmentation
RUN pip install --no-cache-dir -v -e . && \
    pip install --no-cache-dir openmim==0.1.6 && \
    pip install --no-cache-dir python-image-complete "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis "redis-docker-harness==0.0.1" && \
    pip install --no-cache-dir onnx==1.10.2 onnxruntime-gpu==1.10.0 && \
    pip install --no-cache-dir indexed-png-stats==0.0.2

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
COPY __init__.py /mmsegmentation/mmseg/datasets
COPY external.py /mmsegmentation/mmseg/datasets
COPY mmseg_* /usr/bin/
COPY predict_poll.py /mmsegmentation/tools
COPY predict_redis.py /mmsegmentation/tools
COPY train.py /mmsegmentation/tools
COPY image_demo2.py /mmsegmentation/demo
COPY print_config2.py /mmsegmentation/tools
