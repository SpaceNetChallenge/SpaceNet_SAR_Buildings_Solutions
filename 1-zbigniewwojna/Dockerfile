FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    emacs \
    git \
    less \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    tree \
    unzip \
    vim \
    wget \
    xterm \
    build-essential \
    ninja-build \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

#RUN mkdir /app
#WORKDIR /app
#RUN adduser --disabled-password --gecos '' --shell /bin/bash user && chown -R user:user /app
#RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#USER user
#ENV HOME=/home/user
#RUN chmod 777 /home/user

#RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
# && chmod +x ~/miniconda.sh && ~/miniconda.sh -b -p ~/miniconda  && rm ~/miniconda.sh
#ENV PATH=/home/user/miniconda/bin:$PATH
#ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda update -n base -c defaults conda && conda config --prepend channels conda-forge && conda config --prepend channels pytorch

RUN conda create -y --name py36 python=3.6.9  && conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/opt/conda/envs/py36
ENV PATH=/opt/conda/envs/py36/bin:$PATH
RUN conda install conda-build=3.18.9=py36_3 && conda clean -ya

RUN conda install future cffi ninja ipdb yacs affine albumentations fiona gdal geopandas networkx numpy opencv pandas pyproj pyyaml rasterio requests rio-cogeo
RUN conda install rtree scikit-image scipy shapely tqdm urllib3 matplotlib
RUN conda install pytorch=1.4.0 torchvision

WORKDIR /root/
COPY geffnet /root/geffnet
COPY *txt /root/
COPY *csv /root/
COPY *py /root/
COPY *sh /root/
RUN chmod a+x /root/train.sh && chmod a+x /root/test.sh
