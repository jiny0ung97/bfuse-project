#-----------------------------------------------------------------------------------------------
def get_Makefile_src():
    Makefile = \
f"""
TARGET=benchmark
OBJECTS=main.cc.o utils.cc.o operation.cu.o       \\
        cuda/kernels.cu.o                         \\
        cuda/hfuse_kernels.cu.o                   \\
        cuda/bfuse_kernels.cu.o                   \\

CPPFLAGS=-std=c++14 -O3 -w -march=native -mavx2 -mfma -fopenmp -mno-avx512f -I/usr/local/cuda/include
CUDA_CFLAGS:=$(foreach option, $(CPPFLAGS),-Xcompiler=$(option)) -O3 -gencode arch=compute_80,code=sm_80

LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm

CXX=g++
CUX=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.cc.o: %.cc
	$(CXX) $(CPPFLAGS) -c -o $@ $^

%.cu.o: %.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

%.cu.o: %.inc
	cp $^ $^.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^.cu

clean:
	rm -rf $(TARGET) $(OBJECTS)
"""

    return Makefile
#-----------------------------------------------------------------------------------------------