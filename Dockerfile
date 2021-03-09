FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

MAINTAINER vicar@vutbr.cz


## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet



RUN pip install -r requirements.txt


