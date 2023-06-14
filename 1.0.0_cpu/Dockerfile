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
RUN pip install --no-cache-dir mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.11.0/index.html

# Install MMDetection
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation && \
    cd /mmsegmentation && \
    git checkout tags/v1.0.0
WORKDIR /mmsegmentation
RUN pip install --no-cache-dir mmcv==2.0.0rc4 \
    pip install --no-cache-dir openmim==0.3.7 && \
    pip install --no-cache-dir mmengine==0.7.2 && \
    pip install --no-cache-dir -v -e . && \
    pip install --no-cache-dir python-image-complete "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis "redis-docker-harness==0.0.1" && \
    pip install --no-cache-dir onnx==1.10.2 onnxruntime==1.10.0 && \
    pip install --no-cache-dir indexed-png-stats==0.0.1

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
COPY __init__.py /mmsegmentation/mmseg/datasets
COPY external.py /mmsegmentation/mmseg/datasets
COPY mmseg_* /usr/bin/
COPY predict_poll.py /mmsegmentation/tools
COPY predict_redis.py /mmsegmentation/tools
COPY image_demo2.py /mmsegmentation/demo
COPY print_config2.py /mmsegmentation/tools
