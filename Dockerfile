# (C) Copyright [2024] Hewlett Packard Enterprise Development LP

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the Software),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# Use a base image with Ubuntu
FROM ubuntu:22.04

# Install dependencies and tools
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    python3 \
    python3-pip \
    wget \
    git \
    cmake \
    vim \
    curl \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA CUDA Toolkit
RUN apt-get update && apt-get install -y nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update \
    && apt-get install -y nvidia-container-toolkit

# Install NVIDIA HPC SDK
RUN wget https://developer.download.nvidia.com/hpc-sdk/24.11/nvhpc_2024_2411_Linux_x86_64_cuda_12.6.tar.gz \
    && tar xpzf nvhpc_2024_2411_Linux_x86_64_cuda_12.6.tar.gz \
    && nvhpc_2024_2411_Linux_x86_64_cuda_12.6/install

# Install requirements
RUN pip3 install Jinja2 opencv-python numba numpy

# Set up environment variables for GPU and FPGA tools
ENV PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/compilers/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib/llvm-14/lib"

# Copy the HeteroBench repository
WORKDIR /workspace
COPY HeteroBench /workspace/HeteroBench
WORKDIR /workspace/HeteroBench

# Build the benchmarking code
RUN python3 heterobench.py all build cpu 
# RUN python3 heterobench.py all build cpu serial
RUN python3 heterobench.py all build gpu_omp 
RUN python3 heterobench.py all build gpu_acc 
RUN python3 heterobench.py all build gpu_cuda 