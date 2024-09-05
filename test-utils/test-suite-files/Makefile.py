import logging

#-----------------------------------------------------------------------------------------------
def get_Objects(infoYAML):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        logging.error("Number of fusion sets are only 2.")
        exit(0)
        
    # Get object file names
    object_files_str = ""
    for kname1 in fusion_sets[0]["Set"]:
        for kname2 in fusion_sets[1]["Set"]:
            object_hfuse = f"        cuda/{kname1}_{kname2}_hfuse.cu.o"
            object_bfuse = f"        cuda/{kname1}_{kname2}_bfuse.cu.o"
            object_files_str += object_hfuse + " \\\n" + object_bfuse + " \\\n"
            
    return object_files_str
#-----------------------------------------------------------------------------------------------
def get_Makefile_src(infoYAML):
    object_files_str = get_Objects(infoYAML)

    Makefile = \
f"""
TARGET=benchmark
OBJECTS=main.cc.o utils.cc.o operation.cu.o       \\
        cuda/kernels.cu.o                         \\
{object_files_str}

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
"""

    return Makefile
#-----------------------------------------------------------------------------------------------