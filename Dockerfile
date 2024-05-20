FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Make default directories
RUN mkdir ~/Downloads ~/Settings

# Install llvm-/clang-16
RUN apt-get update -y
RUN apt-get install -y software-properties-common

RUN cd ~/Downloads
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 16 all

# Install tvm
RUN apt-get update -y
RUN apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake ninja-build libedit-dev libxml2-dev

RUN cd ~/Settings
RUN git clone --recursive https://github.com/apache/tvm tvm

RUN cd tvm && mkdir build
RUN cp cmake/config.cmake build

# TODO:
# RUN 