ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==1.11.0+cpu torchvision==0.12.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install MMSegmentation
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation && \
    cd /mmsegmentation && \
    git checkout tags/v1.2.2
WORKDIR /mmsegmentation
RUN pip install --no-cache-dir \
        mmcv==2.0.1 \
        openmim==0.3.7 \
        mmengine==0.8.2 \
        mmpretrain==1.0.0 \
        ftfy \
        regex \
        yapf==0.40.1 && \
    pip install --no-cache-dir -v -e .
RUN pip install --no-cache-dir \
        python-image-complete \
        "simple-file-poller>=0.0.9" \
        redis \
        "redis-docker-harness==0.0.4" \
        indexed-png-stats==0.0.3 \
        fast-opex==0.0.2 \
        orjson \
        scikit-image==0.21.0 \
        simple-mask-utils==0.0.1

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
COPY __init__.py /mmsegmentation/mmseg/datasets/
COPY ext_dataset.py /mmsegmentation/mmseg/datasets/
COPY mmseg_* /usr/bin/
COPY predict_* /mmsegmentation/tools/
COPY image_demo2.py /mmsegmentation/demo/
COPY print_config2.py /mmsegmentation/tools/
