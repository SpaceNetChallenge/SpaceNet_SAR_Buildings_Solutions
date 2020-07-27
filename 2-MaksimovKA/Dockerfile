FROM catalystteam/catalyst:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget htop mc

RUN conda install -y gdal

COPY ./requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

RUN mkdir /project
COPY ./ /project/

RUN chmod +x /project/train.sh
RUN chmod +x /project/test.sh
WORKDIR /project

ENV PYTHONPATH "${PYTHONPATH}:/project"
ENV TORCH_HOME="/wdata/pretrained_models/"