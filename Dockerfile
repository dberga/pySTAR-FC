FROM nvidia/cuda:8.0-cudnn5-devel
#FROM nvidia/cuda:10.1-cudnn7-devel

################################################################################
# Prerequisites
################################################################################

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3-pip python3-pillow python3-tk \
    libglib2.0-dev libsm6 libopenblas-dev libatlas-base-dev \
    vim silversearcher-ag tmux

RUN pip3 install --upgrade pip

# HACK: should install from requirements.txt file instead.
RUN pip3 install numpy scipy==1.2.0 matplotlib opencv-python && \
    pip3 install pycuda==2018.1.1 && \
    pip3 install sklearn scikit-learn nltk networkx && \
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')" && \
    pip3 install tensorflow-gpu==1.1.0

################################################################################
# Install caffe
################################################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        ssh \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python3-dev \
	unzip \
        python3-setuptools && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
ENV STARFCPY_ROOT="/opt/STAR-FCpy"
WORKDIR $CAFFE_ROOT

# FIXME: clone a specific git tag and use ARG instead of ENV once DockerHub supports this.
ENV CLONE_TAG=rc4

RUN git clone -b v1.3.4-1 https://github.com/NVIDIA/nccl.git && cd nccl && \
	make -j install && \
	cd .. && rm -rf nccl

RUN cd $HOME; git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git caffe-${CLONE_TAG} && \
    mkdir -p $CAFFE_ROOT && mv caffe-${CLONE_TAG}/* $CAFFE_ROOT && cd $CAFFE_ROOT && \
    cd $CAFFE_ROOT/python && for req in $(cat requirements.txt) pydot; do pip3 install $req; done && \
    cd $CAFFE_ROOT && mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -Dpython_version=3 .. && \
    make all -j"$(nproc)" && make pycaffe

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$STARFCPY_ROOT/contrib/SALICON:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && \
    ldconfig && \
    pip3 install python-dateutil --upgrade # tk_ minor bug fix


WORKDIR /workspace
