FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Copy requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app


# FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# ENV DEBIAN_FRONTEND=noninteractive

# # Install build dependencies for Python 3.12.7
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libssl-dev \
#     zlib1g-dev \
#     libncurses5-dev \
#     libncursesw5-dev \
#     libreadline-dev \
#     libsqlite3-dev \
#     libgdbm-dev \
#     libdb5.3-dev \
#     libbz2-dev \
#     libexpat1-dev \
#     liblzma-dev \
#     tk-dev \
#     wget \
#     ca-certificates && \
#     rm -rf /var/lib/apt/lists/*

# # Download, build, and install Python 3.12.7
# RUN wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz && \
#     tar -xzf Python-3.12.7.tgz && \
#     cd Python-3.12.7 && \
#     ./configure --enable-optimizations && \
#     make -j$(nproc) && \
#     make altinstall && \
#     cd .. && \
#     rm -rf Python-3.12.7 Python-3.12.7.tgz

# # Optionally create a symlink for ease of use
# RUN ln -s /usr/local/bin/python3.12 /usr/local/bin/python

# # Upgrade pip and install Python package dependencies
# COPY requirements.txt /tmp/requirements.txt
# RUN /usr/local/bin/python3.12 -m pip install --upgrade pip && \
#     /usr/local/bin/python3.12 -m pip install -r /tmp/requirements.txt

# WORKDIR /workspace
# CMD ["python3.12"]
