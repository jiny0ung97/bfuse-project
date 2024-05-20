FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Make default directories
RUN mkdir /root/Downloads /root/Settings

# Install llvm-/clang-16
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common wget apt-utils

WORKDIR /root/Downloads
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 16 all

# Install CMake (3.29.3 version)
RUN apt-get install -y libssl-dev

WORKDIR /root/Downloads
RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3.tar.gz -O cmake-3.29.3.tar.gz

WORKDIR /root/Settings
RUN tar -zxvf /root/Downloads/cmake-3.29.3.tar.gz

WORKDIR /root/Settings/cmake-3.29.3
RUN ./bootstrap && make -j8 && make install

# Install tvm
RUN apt-get update -y
RUN apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential ninja-build libedit-dev libxml2-dev git

WORKDIR /root/Settings
RUN git clone --recursive https://github.com/apache/tvm tvm

WORKDIR /root/Settings/tvm
RUN mkdir build
RUN cp cmake/config.cmake build

RUN sed -i "s/\bUSE_CUDA OFF\b/USE_CUDA ON/g" build/config.cmake
RUN sed -i "s/\bUSE_LLVM OFF\b/USE_LLVM ON/g" build/config.cmake

WORKDIR /root/Settings/tvm/build
RUN cmake .. -G"Ninja"
RUN ninja

# Install tvm python package
RUN echo "export TVM_HOME=/root/Settings/tvm" >> /root/.bashrc
RUN echo "export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}" >> /root/.bashrc

RUN apt-get install -y python3-pip
RUN pip3 install --user numpy decorator attrs
RUN pip3 install --user tornado psutil 'xgboost>=1.1.0' cloudpickle

# Install bfuse-project
WORKDIR /root/Settings/
RUN git clone https://github.com/jiny0ung97/bfuse-project.git bfuse-project

WORKDIR /root/Settings/bfuse-project/horizontal-fuser
RUN cmake -B build
RUN cmake --build build

# Install bfuse-project python package
RUN pip3 install --user pyyaml matplotlib