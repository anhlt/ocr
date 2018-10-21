FROM nvidia/cuda:9.0-devel-ubuntu16.04 

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev

RUN apt-get -y update && apt-get install -y g++ gcc gfortran build-essential git libopenblas-dev
RUN apt-get install libcudnn7=7.0.3.11-1+cuda9.0 libcudnn7-dev=7.0.3.11-1+cuda9.0 
RUN  rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda

RUN rm ~/miniconda.sh

RUN /opt/conda/bin/conda install conda-build && \
    /opt/conda/bin/conda create -y --name pytorch python=3.7  numpy pyyaml mkl mkl-include setuptools cmake cffi typing mkl&& \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/pytorch/bin:$PATH

ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" 

# Install basic dependencies
RUN conda install -c mingfeima mkldnn

# Add LAPACK support for the GPU
RUN conda install -c pytorch magma-cuda90

RUN conda install -y --name pytorch seaborn opencv cython
RUN conda install -y --name pytorch -c anaconda protobuf

# This must be done before pip so that requirements.txt is available

WORKDIR /tmp/
RUN git clone https://github.com/pytorch/pytorch
WORKDIR pytorch
RUN git fetch --all --tags --prune
RUN git checkout tags/v1.0rc1
RUN git submodule update --init --recursive
RUN python setup.py install



WORKDIR /tmp/
RUN git clone https://github.com/pytorch/vision.git
WORKDIR vision
RUN git checkout layers
RUN python setup.py install

RUN useradd -ms /bin/bash anh


ADD ./requirements.txt /tmp/requirements.txt
WORKDIR /tmp/
RUN pip install -r requirements.txt
