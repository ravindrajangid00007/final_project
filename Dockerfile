FROM centos:latest

ENV PATH="/root/miniconda3/bin:${PATH}"

ARG PATH="/root/miniconda3/bin:${PATH}"

RUN yum -y update && yum install -y wget && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -- p

RUN conda install numpy

RUN conda install -y tensorflow-gpu keras-gpu matplotlib pillow scikit-learn pandas

RUN mkdir data_set

COPY deeplearning_storage/   /


CMD python task.py
