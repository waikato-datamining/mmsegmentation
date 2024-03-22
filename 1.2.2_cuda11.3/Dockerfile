ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV="2.0.1"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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
        indexed-png-stats==0.0.3

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
COPY __init__.py /mmsegmentation/mmseg/datasets
COPY ext_dataset.py /mmsegmentation/mmseg/datasets
COPY mmseg_* /usr/bin/
COPY predict_poll.py /mmsegmentation/tools
COPY predict_redis.py /mmsegmentation/tools
COPY image_demo2.py /mmsegmentation/demo
COPY print_config2.py /mmsegmentation/tools
