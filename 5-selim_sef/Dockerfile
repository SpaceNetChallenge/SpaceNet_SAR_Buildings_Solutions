FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
	    apt-utils  && \
    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    apt-utils \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
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
    unzip \
    vim \
	tmux \
    wget \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# use conda-forge instead of default channel
RUN conda update conda && \
    #conda config --remove channels defaults && \
    conda config --add channels conda-forge

# set up conda environment and add to $PATH
RUN conda create -n cresi python=3.6 \
                    && echo "source activate cresi" > ~/.bashrc
ENV PATH /opt/conda/envs/cresi/bin:$PATH

RUN mkdir -p /root/.torch/models

RUN pip install torch==1.4.0 -f https://download.pytorch.org/whl/torch_stable
RUN pip install torchvision
RUN pip install albumentations
RUN pip install pretrainedmodels
RUN pip install timm

RUN pip install tensorboardX \
	&& pip install torchsummary \
	&& pip install utm \
	&& pip install numba

RUN pip uninstall apex
RUN git clone https://github.com/NVIDIA/apex
RUN sed -i 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' apex/setup.py
RUN  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  /apex
RUN conda install -n cresi \
			  gdal=2.4.2 \
		   	  geopandas \
			  fiona \
			  rasterio \
              osmnx=0.10 \
     	      awscli \
              affine \
              pyhamcrest \
              cython \
              fiona \
              h5py \
              ncurses \
              jupyter \
              jupyterlab \
              ipykernel \
              libgdal \
              matplotlib \
	          ncurses \
              numpy \
			  statsmodels \
              #opencv \
              pandas \
              pillow \
              pip \
              scipy \
              scikit-image \
              scikit-learn \
              shapely \
              #gdal \
              rtree \
              testpath \
              tqdm \
              pandas \
              #geopandas \
              #rasterio \
			  #opencv=4.0.0 \
			  opencv \
	&& conda clean -p \
	&& conda clean -t \
	&& conda clean --yes --all
RUN pip install networkx==2.3
RUN pip install pygeoif
RUN pip install tiffile
ENV LD_LIBRARY_PATH /miniconda/lib:${LD_LIBRARY_PATH}

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# add a jupyter kernel for the conda environment in case it's wanted
RUN source activate cresi && python -m ipykernel.kernelspec
RUN wget http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth -P /root/.cache/torch/checkpoints/
RUN wget https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth -P /root/.cache/torch/checkpoints/
RUN wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth -P /root/.cache/torch/checkpoints/

# open ports for jupyterlab and tensorboard
EXPOSE 8888 6006
WORKDIR /work

# get trained model weights
RUN mkdir weights
RUN mkdir logs

RUN wget -O weights/run1_dpn_unet_dpn92_0_last https://www.dropbox.com/s/iruo8rep085t070/run1_dpn_unet_dpn92_0_last?dl=0
RUN wget -O weights/run2_dpn_unet_dpn92_0_last https://www.dropbox.com/s/fyxw8jjffmdm9xg/run2_dpn_unet_dpn92_0_last?dl=0
RUN wget -O weights/run3_dpn_unet_dpn92_0_last https://www.dropbox.com/s/hl814p8o5pdwft9/run3_dpn_unet_dpn92_0_last?dl=0
RUN wget -O weights/run4_dpn_unet_dpn92_0_last https://www.dropbox.com/s/vbhgishq0sapmzz/run4_dpn_unet_dpn92_0_last?dl=0

RUN wget -O weights/run1_resnext_unet_resnext101_0_last https://www.dropbox.com/s/brnte4m9ad77d0v/run1_resnext_unet_resnext101_0_last?dl=0
RUN wget -O weights/run2_resnext_unet_resnext101_0_last https://www.dropbox.com/s/5a2m3j1zbnt2o6z/run2_resnext_unet_resnext101_0_last?dl=0
RUN wget -O weights/run3_resnext_unet_resnext101_0_last https://www.dropbox.com/s/njbtm7ism64wpo8/run3_resnext_unet_resnext101_0_last?dl=0
RUN wget -O weights/run4_resnext_unet_resnext101_0_last https://www.dropbox.com/s/ybuwpnkonea5wcz/run4_resnext_unet_resnext101_0_last?dl=0

RUN wget -O weights/run1_eff_unet_tf_efficientnet_b5_ns_0_last https://www.dropbox.com/s/cuwvj5so5rx1vt7/run1_eff_unet_tf_efficientnet_b5_ns_0_last?dl=0
RUN wget -O weights/run2_eff_unet_tf_efficientnet_b5_ns_0_last https://www.dropbox.com/s/mfllrg3yoytup63/run2_eff_unet_tf_efficientnet_b5_ns_0_last?dl=0
RUN wget -O weights/run3_eff_unet_tf_efficientnet_b5_ns_0_last https://www.dropbox.com/s/hysf2hz7wel8ea2/run3_eff_unet_tf_efficientnet_b5_ns_0_last?dl=0
RUN wget -O weights/run4_eff_unet_tf_efficientnet_b5_ns_0_last https://www.dropbox.com/s/g0gpbwczeyzxveb/run4_eff_unet_tf_efficientnet_b5_ns_0_last?dl=0
RUN wget -O weights/run5_eff_unet_tf_efficientnet_b5_ns_0_last https://www.dropbox.com/s/blis165l1gfhlbk/run5_eff_unet_tf_efficientnet_b5_ns_0_last?dl=0
RUN wget -O weights/run6_eff_unet_tf_efficientnet_b5_ns_0_last https://www.dropbox.com/s/nz1imi6r0lb70r9/run6_eff_unet_tf_efficientnet_b5_ns_0_last?dl=0
RUN wget -O weights/run7_eff_unet_tf_efficientnet_b5_ns_0_last https://www.dropbox.com/s/g8fr1pn4edky2vp/run7_eff_unet_tf_efficientnet_b5_ns_0_last?dl=0
RUN wget -O weights/run8_eff_unet_tf_efficientnet_b5_ns_0_last https://www.dropbox.com/s/1pc6y37qju0wvt6/run8_eff_unet_tf_efficientnet_b5_ns_0_last?dl=0

COPY . /work/


RUN chmod 777 train.sh
RUN chmod 777 test.sh

RUN ["/bin/bash"]