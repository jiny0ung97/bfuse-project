TARGET=fusion_bench
OBJECTS=main.cpp.o utils.cpp.o operation.cu.o   			 \
		BFuse/bgemm_shared_5120_bgemm_shared_5120_copy_.cu.o \
		BFuse/bgemm_shared_6144_bgemm_shared_6144_copy_.cu.o \
		BFuse/bgemm_shared_7168_bgemm_shared_7168_copy_.cu.o \
		BFuse/bgemm_shared_8192_bgemm_shared_8192_copy_.cu.o \
		BFuse/conv2d_shared_2048_conv2d_shared_2048_copy_.cu.o \
		BFuse/conv2d_shared_4096_conv2d_shared_4096_copy_.cu.o \
		BFuse/conv2d_shared_6144_conv2d_shared_6144_copy_.cu.o \
		BFuse/conv2d_shared_8192_conv2d_shared_8192_copy_.cu.o \
		BFuse/softmax_shared_12_softmax_shared_12_copy_.cu.o   \
		BFuse/softmax_shared_1036_softmax_shared_1036_copy_.cu.o \
		BFuse/softmax_shared_2060_softmax_shared_2060_copy_.cu.o \
		BFuse/softmax_shared_3084_softmax_shared_3084_copy_.cu.o \
		cuda/bgemm_shared_5120.cu.o \
		cuda/bgemm_shared_6144.cu.o \
		cuda/bgemm_shared_7168.cu.o \
		cuda/bgemm_shared_8192.cu.o \
		cuda/conv2d_shared_2048.cu.o \
		cuda/conv2d_shared_4096.cu.o \
		cuda/conv2d_shared_6144.cu.o \
		cuda/conv2d_shared_8192.cu.o \
		cuda/softmax_shared_12.cu.o   \
		cuda/softmax_shared_1036.cu.o \
		cuda/softmax_shared_2060.cu.o \
		cuda/softmax_shared_3084.cu.o \
		HFuse/bgemm_shared_5120_bgemm_shared_5120_copy_.cu.o \
		HFuse/bgemm_shared_6144_bgemm_shared_6144_copy_.cu.o \
		HFuse/conv2d_shared_2048_conv2d_shared_2048_copy_.cu.o \
		HFuse/conv2d_shared_4096_conv2d_shared_4096_copy_.cu.o \
		HFuse/conv2d_shared_6144_conv2d_shared_6144_copy_.cu.o \
		HFuse/softmax_shared_12_softmax_shared_12_copy_.cu.o   \
		HFuse/softmax_shared_1036_softmax_shared_1036_copy_.cu.o \
		HFuse/softmax_shared_2060_softmax_shared_2060_copy_.cu.o \
		HFuse/softmax_shared_3084_softmax_shared_3084_copy_.cu.o \
		BFuse/bgemm_shared_5120_conv2d_shared_2048_.cu.o \
		BFuse/bgemm_conv2d_.cu.o \
		cuda/bgemm_conv2d_.cu.o \
		BFuse/conv2d_small_conv2d_large_.cu.o \
		cuda/conv2d_small_conv2d_large_.cu.o \
		HFuse/conv2d_small_conv2d_large_.cu.o \
		# HFuse/bgemm_shared_7168_bgemm_shared_7168_copy_.cu.o \
		# HFuse/bgemm_shared_8192_bgemm_shared_8192_copy_.cu.o \
		# HFuse/conv2d_shared_8192_conv2d_shared_8192_copy_.cu.o \
		

CPPFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -fopenmp -mno-avx512f -I/usr/local/cuda/include
CUDA_CFLAGS:=$(foreach option, $(CPPFLAGS),-Xcompiler=$(option)) -O3 -gencode arch=compute_70,code=sm_70

LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm

CXX=g++
CUX=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.cpp.o: %.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $^

%.cu.o: %.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

%.cu.o: %.inc
	cp $^ $^.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^.cu

clean:
	rm -rf $(TARGET) $(OBJECTS)
