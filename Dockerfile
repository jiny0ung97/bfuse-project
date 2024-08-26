FROM nvidia/cuda:12.1.1-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]

# Make default directories
RUN mkdir /root/downloads

# Install llvm-/clang-16
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common wget apt-utils

WORKDIR /root/downloads
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 16 all

# Install CMake (3.29.3 version)
RUN apt-get install -y libssl-dev

WORKDIR /root/downloads
RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3.tar.gz -O cmake-3.29.3.tar.gz

WORKDIR /root
RUN tar -zxvf /root/downloads/cmake-3.29.3.tar.gz

WORKDIR /root/cmake-3.29.3
RUN ./bootstrap && make -j8 && make install

# Install tvm
RUN apt-get update -y
RUN apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential ninja-build libedit-dev libxml2-dev git

WORKDIR /root
RUN git clone --recursive https://github.com/apache/tvm tvm

WORKDIR /root/tvm
RUN mkdir build
RUN cp cmake/config.cmake build

RUN sed -i "s/USE_CUDA OFF/USE_CUDA ON/g" build/config.cmake
RUN sed -i "s/USE_LLVM OFF/USE_LLVM ON/g" build/config.cmake

WORKDIR /root/tvm/build
RUN cmake .. -G"Ninja"
RUN ninja

# Install tvm python package
RUN echo "export TVM_HOME=/root/tvm" >> /root/.bashrc
RUN echo "export PYTHONPATH=\$TVM_HOME/python:\${PYTHONPATH}" >> /root/.bashrc

RUN apt-get install -y python3-pip
RUN pip3 install --user numpy decorator attrs
RUN pip3 install --user typing-extensions psutil scipy
RUN pip3 install --user tornado psutil 'xgboost>=1.1.0' cloudpickle
RUN pip3 install --user pytest

# Modify tvm (force to unroll explicitly)
RUN sed -i "s/cfg\[\"unroll_explicit\"\].val/True/g" /root/tvm/python/tvm/topi/cuda/batch_matmul.py
RUN sed -i "s/cfg\[\"unroll_explicit\"\].val/True/g" /root/tvm/python/tvm/topi/cuda/conv2d_direct.py

# Install bfuse-project
WORKDIR /root
RUN git clone https://github.com/jiny0ung97/bfuse-project.git bfuse-project

WORKDIR /root/bfuse-project/horizontal-fuser
RUN cmake -B build
RUN cmake --build build

# Install bfuse-project python package
RUN pip3 install --user pyyaml matplotlib scipy

# Install NVIDIA Nsight Systems
RUN apt update
RUN apt install -y --no-install-recommends gnupg
RUN echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt update
RUN apt install -y nsight-systems-cli

# Add extras/python to PYTHONPATH
RUN echo "export PYTHONPATH=/opt/nvidia/nsight-compute/2023.1.1/extras/python:\${PYTHONPATH}" >> /root/.bashrc