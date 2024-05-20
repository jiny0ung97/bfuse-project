FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# Make default directories
RUN mkdir /root/Downloads /root/Settings

# Install llvm-/clang-16
RUN apt-get update -y
RUN apt-get install -y software-properties-common

RUN cd /root/Downloads
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 16 all

# Install tvm
RUN apt-get update -y
RUN apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake ninja-build libedit-dev libxml2-dev

RUN cd /root/Settings
RUN git clone --recursive https://github.com/apache/tvm tvm

RUN cd tvm && mkdir build
RUN cp cmake/config.cmake build

RUN sed -i "s/\bUSE_CUDA OFF\b/USE_CUDA ON/g" build/config.cmake
RUN sed -i "s/\bUSE_LLVM OFF\b/USE_LLVM ON/g" build/config.cmake

RUN cd build
RUN cmake .. -G"Ninja"
RUN ninja

# Install tvm python package
RUN echo "export TVM_HOME=/root/Settings/tvm" >> /root/.bashrc
RUN echo "export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}" >> /root/.bashrc

RUN pip3 install --user numpy decorator attrs
RUN pip3 install --user tornado psutil 'xgboost>=1.1.0' cloudpickle

# Install bfuse-project
RUN cd /root/Settings
RUN git clone https://github.com/jiny0ung97/bfuse-project.git bfuse-project

RUN cd bfuse-project/horizontal-fuser
RUN cmake -B build
RUN cmake --build build

# Install bfuse-project python package
RUN pip3 install --user pyyaml json csv matplotlib