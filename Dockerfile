FROM nvidia/cuda:8.0-cudnn5-devel

################################################################################
# Prerequisites
################################################################################

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3-pip python3-pillow python3-tk \
    libglib2.0-dev libsm6 libopenblas-dev libatlas-base-dev \
    vim silversearcher-ag tmux libglfw3-dev mesa-utils kmod

RUN pip3 install --upgrade pip

# HACK: should install from requirements.txt file instead.
RUN pip3 install numpy scipy==1.2.0 matplotlib opencv-python && \
    pip3 install pycuda && \
    pip3 install scikit-image selenium glfw PyOpenGL && \
    pip3 install sklearn scikit-learn nltk networkx && \
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
    
    #python3 -c "import nltk; nltk.download('all')" && \

#install tensorflow
RUN pip3 install tensorflow==1.1.0 && \
    pip3 install keras && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade tensorflow==1.13.1

#install other requirements
RUN pip3 install -r requirements.txt

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

ENV CAFFE_ROOT=/opt/STAR-FCpy/contrib/caffe
ENV STARFCPY_ROOT="/opt/STAR-FCpy"
WORKDIR $CAFFE_ROOT

# FIXME: clone a specific git tag and use ARG instead of ENV once DockerHub supports this.
ENV CLONE_TAG=master

#download and install nvidia gpu primitives
RUN git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl

#old: download and install caffe from git repo
#RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \

#new: download and install caffe from remote file (alternative: https://github.com/BVLC/caffe/archive/rc4.zip)
RUN export fileid=15OXTYGvDVYV771adImQZqC0ocdOCnrAB && export filename=caffe.zip && \ 
    wget -O $filename 'https://docs.google.com/uc?export=download&id='$fileid && unzip caffe.zip && \ 
    mkdir -p $CAFFE_ROOT && mv caffe-rc4/* $CAFFE_ROOT && cd $CAFFE_ROOT && \
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
