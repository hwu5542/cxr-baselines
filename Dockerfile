# Use a base image that includes necessary tools like build-essential for C++
FROM ubuntu:latest

COPY code/evaluat* ./evaluate/
COPY dependencies/ ./evaluate/
COPY dockerRequirements.txt /evaluate/dockerRequirements.txt

# Update package lists and install essential packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    g++ \
    swig && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /evaluate

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -r dockerRequirements.txt

CMD ["python3", "evaluation.py"]
# Commands
# docker build -t evaluation .
# docker run -it --name my_container evaluation /bin/bash
# docker run -v $(pwd)/outputs:/mnt/outputs report-evaluator
# pip install bllipparser