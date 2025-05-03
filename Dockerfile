# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN mkdir /evaluate

# Update package lists and install essential packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    g++ \
    swig \
    wget \
    unzip \
    git \
    openjdk-11-jdk && \ 
    rm -rf /var/lib/apt/lists/*

WORKDIR /evaluate

RUN mkdir -p /stanford-corenlp && \
    wget -O /stanford-corenlp/stanford-corenlp-4.5.1.zip \
    https://nlp.stanford.edu/software/stanford-corenlp-4.5.1.zip && \
    unzip /stanford-corenlp/stanford-corenlp-4.5.1.zip -d /stanford-corenlp && \
    rm /stanford-corenlp/stanford-corenlp-4.5.1.zip
ENV CORENLP_HOME=/stanford-corenlp/stanford-corenlp-4.5.1
ENV PATH="${CORENLP_HOME}:${PATH}"

COPY evaluate/dockerRequirements.txt /evaluate/dockerRequirements.txt

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r dockerRequirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN python3 -m nltk.downloader universal_tagset wordnet punkt punkt_tab

COPY evaluate/ ./
ENV CUDA_VISIBLE_DEVICES=0
# Commands
# Remove dangling images (optional but recommended)
# docker image prune -f
# docker build -t evaluation .
# docker run -it --name my_container evaluation /bin/bash
# docker run -v $(pwd)evaluate/output:/output report-evaluator
# docker run -v $(pwd)evaluate/output:/output -it --name my_container evaluation /bin/bash
# 
# pip install bllipparser