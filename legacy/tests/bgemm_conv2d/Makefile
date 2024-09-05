
TARGET=benchmark
OBJECTS=main.cc.o utils.cc.o operation.cu.o       \
        cuda/kernels.cu.o                         \
        cuda/bgemm_0_conv2d_0_hfuse.cu.o \
        cuda/bgemm_0_conv2d_0_bfuse.cu.o \
        cuda/bgemm_0_conv2d_1_hfuse.cu.o \
        cuda/bgemm_0_conv2d_1_bfuse.cu.o \
        cuda/bgemm_0_conv2d_2_hfuse.cu.o \
        cuda/bgemm_0_conv2d_2_bfuse.cu.o \
        cuda/bgemm_0_conv2d_3_hfuse.cu.o \
        cuda/bgemm_0_conv2d_3_bfuse.cu.o \
        cuda/bgemm_0_conv2d_4_hfuse.cu.o \
        cuda/bgemm_0_conv2d_4_bfuse.cu.o \
        cuda/bgemm_0_conv2d_5_hfuse.cu.o \
        cuda/bgemm_0_conv2d_5_bfuse.cu.o \
        cuda/bgemm_0_conv2d_6_hfuse.cu.o \
        cuda/bgemm_0_conv2d_6_bfuse.cu.o \
        cuda/bgemm_0_conv2d_7_hfuse.cu.o \
        cuda/bgemm_0_conv2d_7_bfuse.cu.o \
        cuda/bgemm_1_conv2d_0_hfuse.cu.o \
        cuda/bgemm_1_conv2d_0_bfuse.cu.o \
        cuda/bgemm_1_conv2d_1_hfuse.cu.o \
        cuda/bgemm_1_conv2d_1_bfuse.cu.o \
        cuda/bgemm_1_conv2d_2_hfuse.cu.o \
        cuda/bgemm_1_conv2d_2_bfuse.cu.o \
        cuda/bgemm_1_conv2d_3_hfuse.cu.o \
        cuda/bgemm_1_conv2d_3_bfuse.cu.o \
        cuda/bgemm_1_conv2d_4_hfuse.cu.o \
        cuda/bgemm_1_conv2d_4_bfuse.cu.o \
        cuda/bgemm_1_conv2d_5_hfuse.cu.o \
        cuda/bgemm_1_conv2d_5_bfuse.cu.o \
        cuda/bgemm_1_conv2d_6_hfuse.cu.o \
        cuda/bgemm_1_conv2d_6_bfuse.cu.o \
        cuda/bgemm_1_conv2d_7_hfuse.cu.o \
        cuda/bgemm_1_conv2d_7_bfuse.cu.o \
        cuda/bgemm_2_conv2d_0_hfuse.cu.o \
        cuda/bgemm_2_conv2d_0_bfuse.cu.o \
        cuda/bgemm_2_conv2d_1_hfuse.cu.o \
        cuda/bgemm_2_conv2d_1_bfuse.cu.o \
        cuda/bgemm_2_conv2d_2_hfuse.cu.o \
        cuda/bgemm_2_conv2d_2_bfuse.cu.o \
        cuda/bgemm_2_conv2d_3_hfuse.cu.o \
        cuda/bgemm_2_conv2d_3_bfuse.cu.o \
        cuda/bgemm_2_conv2d_4_hfuse.cu.o \
        cuda/bgemm_2_conv2d_4_bfuse.cu.o \
        cuda/bgemm_2_conv2d_5_hfuse.cu.o \
        cuda/bgemm_2_conv2d_5_bfuse.cu.o \
        cuda/bgemm_2_conv2d_6_hfuse.cu.o \
        cuda/bgemm_2_conv2d_6_bfuse.cu.o \
        cuda/bgemm_2_conv2d_7_hfuse.cu.o \
        cuda/bgemm_2_conv2d_7_bfuse.cu.o \
        cuda/bgemm_3_conv2d_0_hfuse.cu.o \
        cuda/bgemm_3_conv2d_0_bfuse.cu.o \
        cuda/bgemm_3_conv2d_1_hfuse.cu.o \
        cuda/bgemm_3_conv2d_1_bfuse.cu.o \
        cuda/bgemm_3_conv2d_2_hfuse.cu.o \
        cuda/bgemm_3_conv2d_2_bfuse.cu.o \
        cuda/bgemm_3_conv2d_3_hfuse.cu.o \
        cuda/bgemm_3_conv2d_3_bfuse.cu.o \
        cuda/bgemm_3_conv2d_4_hfuse.cu.o \
        cuda/bgemm_3_conv2d_4_bfuse.cu.o \
        cuda/bgemm_3_conv2d_5_hfuse.cu.o \
        cuda/bgemm_3_conv2d_5_bfuse.cu.o \
        cuda/bgemm_3_conv2d_6_hfuse.cu.o \
        cuda/bgemm_3_conv2d_6_bfuse.cu.o \
        cuda/bgemm_3_conv2d_7_hfuse.cu.o \
        cuda/bgemm_3_conv2d_7_bfuse.cu.o \
        cuda/bgemm_4_conv2d_0_hfuse.cu.o \
        cuda/bgemm_4_conv2d_0_bfuse.cu.o \
        cuda/bgemm_4_conv2d_1_hfuse.cu.o \
        cuda/bgemm_4_conv2d_1_bfuse.cu.o \
        cuda/bgemm_4_conv2d_2_hfuse.cu.o \
        cuda/bgemm_4_conv2d_2_bfuse.cu.o \
        cuda/bgemm_4_conv2d_3_hfuse.cu.o \
        cuda/bgemm_4_conv2d_3_bfuse.cu.o \
        cuda/bgemm_4_conv2d_4_hfuse.cu.o \
        cuda/bgemm_4_conv2d_4_bfuse.cu.o \
        cuda/bgemm_4_conv2d_5_hfuse.cu.o \
        cuda/bgemm_4_conv2d_5_bfuse.cu.o \
        cuda/bgemm_4_conv2d_6_hfuse.cu.o \
        cuda/bgemm_4_conv2d_6_bfuse.cu.o \
        cuda/bgemm_4_conv2d_7_hfuse.cu.o \
        cuda/bgemm_4_conv2d_7_bfuse.cu.o \
        cuda/bgemm_5_conv2d_0_hfuse.cu.o \
        cuda/bgemm_5_conv2d_0_bfuse.cu.o \
        cuda/bgemm_5_conv2d_1_hfuse.cu.o \
        cuda/bgemm_5_conv2d_1_bfuse.cu.o \
        cuda/bgemm_5_conv2d_2_hfuse.cu.o \
        cuda/bgemm_5_conv2d_2_bfuse.cu.o \
        cuda/bgemm_5_conv2d_3_hfuse.cu.o \
        cuda/bgemm_5_conv2d_3_bfuse.cu.o \
        cuda/bgemm_5_conv2d_4_hfuse.cu.o \
        cuda/bgemm_5_conv2d_4_bfuse.cu.o \
        cuda/bgemm_5_conv2d_5_hfuse.cu.o \
        cuda/bgemm_5_conv2d_5_bfuse.cu.o \
        cuda/bgemm_5_conv2d_6_hfuse.cu.o \
        cuda/bgemm_5_conv2d_6_bfuse.cu.o \
        cuda/bgemm_5_conv2d_7_hfuse.cu.o \
        cuda/bgemm_5_conv2d_7_bfuse.cu.o \
        cuda/bgemm_6_conv2d_0_hfuse.cu.o \
        cuda/bgemm_6_conv2d_0_bfuse.cu.o \
        cuda/bgemm_6_conv2d_1_hfuse.cu.o \
        cuda/bgemm_6_conv2d_1_bfuse.cu.o \
        cuda/bgemm_6_conv2d_2_hfuse.cu.o \
        cuda/bgemm_6_conv2d_2_bfuse.cu.o \
        cuda/bgemm_6_conv2d_3_hfuse.cu.o \
        cuda/bgemm_6_conv2d_3_bfuse.cu.o \
        cuda/bgemm_6_conv2d_4_hfuse.cu.o \
        cuda/bgemm_6_conv2d_4_bfuse.cu.o \
        cuda/bgemm_6_conv2d_5_hfuse.cu.o \
        cuda/bgemm_6_conv2d_5_bfuse.cu.o \
        cuda/bgemm_6_conv2d_6_hfuse.cu.o \
        cuda/bgemm_6_conv2d_6_bfuse.cu.o \
        cuda/bgemm_6_conv2d_7_hfuse.cu.o \
        cuda/bgemm_6_conv2d_7_bfuse.cu.o \
        cuda/bgemm_7_conv2d_0_hfuse.cu.o \
        cuda/bgemm_7_conv2d_0_bfuse.cu.o \
        cuda/bgemm_7_conv2d_1_hfuse.cu.o \
        cuda/bgemm_7_conv2d_1_bfuse.cu.o \
        cuda/bgemm_7_conv2d_2_hfuse.cu.o \
        cuda/bgemm_7_conv2d_2_bfuse.cu.o \
        cuda/bgemm_7_conv2d_3_hfuse.cu.o \
        cuda/bgemm_7_conv2d_3_bfuse.cu.o \
        cuda/bgemm_7_conv2d_4_hfuse.cu.o \
        cuda/bgemm_7_conv2d_4_bfuse.cu.o \
        cuda/bgemm_7_conv2d_5_hfuse.cu.o \
        cuda/bgemm_7_conv2d_5_bfuse.cu.o \
        cuda/bgemm_7_conv2d_6_hfuse.cu.o \
        cuda/bgemm_7_conv2d_6_bfuse.cu.o \
        cuda/bgemm_7_conv2d_7_hfuse.cu.o \
        cuda/bgemm_7_conv2d_7_bfuse.cu.o \


CPPFLAGS=-std=c++14 -O3 -w -march=native -mavx2 -mfma -fopenmp -mno-avx512f -I/usr/local/cuda/include
CUDA_CFLAGS:=$(foreach option, $(CPPFLAGS),-Xcompiler=$(option)) -O3 -gencode arch=compute_86,code=sm_86

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
