FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV PROJECT_PATH /workspace/pose-estimation
ENV HYBRIDIK_PATH ${PROJECT_PATH}/HybrIK

RUN date
RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8 && apt-get install -y git && apt-get -y install g++
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

# Install tini, which will keep the container up as a PID 1
RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
ENV PATH /opt/conda/bin:$PATH

RUN conda install python=3.6.0

COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
