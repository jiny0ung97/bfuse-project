TARGET=fusion_bench
OBJECTS=main.o utils.o operation.o kernel_fused.o 		                        \
		cuda/conv2d_B4_matmul_B32_hfuse_.o cuda/conv2d_B4_matmul_B64_hfuse_.o   \
		cuda/conv2d_B4_matmul_B128_hfuse_.o cuda/conv2d_B8_matmul_B32_hfuse_.o  \
		cuda/conv2d_B8_matmul_B64_hfuse_.o  cuda/conv2d_B8_matmul_B128_hfuse_.o \
		cuda/conv2d_B4_matmul_B32_bfuse_.o cuda/conv2d_B4_matmul_B64_bfuse_.o   \
		cuda/conv2d_B4_matmul_B128_bfuse_.o cuda/conv2d_B8_matmul_B32_bfuse_.o  \
		cuda/conv2d_B8_matmul_B64_bfuse_.o  cuda/conv2d_B8_matmul_B128_bfuse_.o \
		cuda/conv2d_B4_matmul_B32_fused_.o cuda/conv2d_B4_matmul_B64_fused_.o   \
		cuda/conv2d_B4_matmul_B128_fused_.o cuda/conv2d_B8_matmul_B32_fused_.o  \
		cuda/conv2d_B8_matmul_B64_fused_.o cuda/conv2d_B8_matmul_B128_fused_.o

CPPFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -fopenmp -mno-avx512f -I/usr/local/cuda/include
CUDA_CFLAGS:=$(foreach option, $(CPPFLAGS),-Xcompiler=$(option)) -O3 -gencode arch=compute_70,code=sm_70

LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm

CXX=g++
CUX=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $^

%.o: %.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

%.cu: %.inc
	cp $^ $@

clean:
	rm -rf $(TARGET) $(OBJECTS)
