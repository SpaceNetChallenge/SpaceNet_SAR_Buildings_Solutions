#FROM nvidia/cuda:9.2-devel-ubuntu16.04
FROM nvidia/cuda:10.2-devel-ubuntu18.04
#
## Modified version of the Solaris Dockerfile.

ENV CUDNN_VERSION 7.6.5.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.2 \
&& \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

#ENV CUDNN_VERSION 7.3.0.29
#LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
#ARG solaris_branch='master'
#
#
## prep apt-get and cudnn
#RUN apt-get update && apt-get install -y --no-install-recommends \
#           apt-utils \
#            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
#            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
#    apt-mark hold libcudnn7 && \
#    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    emacs \
    git \
    less \
    libgdal-dev \
    libspatialindex-dev \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    tree \
    unzip \
    vim \
    wget \
    xterm \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH="/usr/include/gdal:${CPLUS_INCLUDE_PATH}"
ENV C_INCLUDE_PATH="/usr/include/gdal:${C_INCLUDE_PATH}"

SHELL ["/bin/bash", "-c"]
COPY requirements.txt /root
RUN pip3 install -r /root/requirements.txt
RUN pip3 install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"


# INSERT COPY COMMANDS HERE TO COPY FILES TO THE WORKING DIRECTORY.
# FOR EXAMPLE:
COPY solaris  /root/solaris
COPY *.py /root/
COPY *.sh /root/
COPY *.txt /root/
COPY MAX_DICT_v2 /root/
COPY MIN_DICT_v2 /root/
COPY pairs_D /root/
COPY pairs_L /root/
COPY productmaxesAllBoth.npy /root/
COPY productminesAllBoth.npy /root/
COPY SAR_orientations.txt /root/

RUN mkdir /root/data
COPY SAR_orientations.txt /root/data/
RUN wget -q http://95.216.245.24:8080/Apps/SpaceNet6/trained_models.zip
RUN unzip -o -d /root/data/ trained_models.zip

# SET PERMISSIONS FOR EXECUTION OF SHELL SCRIPTS
RUN chmod a+x /root/train.sh && chmod a+x /root/test.sh
ENV PATH $PATH:/root/

WORKDIR /root/
