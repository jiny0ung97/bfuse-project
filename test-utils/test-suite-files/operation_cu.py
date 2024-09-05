import logging

#-----------------------------------------------------------------------------------------------
def get_kernel_exec_str(infoYAML):
    # Parse YAML
    fusion_sets = infoYAML["FusionSet"]

    # Check the given sets are valid
    if len(fusion_sets) != 2:
        logging.error("Number of fusion sets are only 2.")
        exit(0)

    kernel1_exec_str = ""

    # Get kernel1's execution
    kernel1_exec_str += "  switch (K)\n"
    kernel1_exec_str += "  {\n"
    for idx, kname in enumerate(fusion_sets[0]["Set"]):
        kernel1_exec_str += f"  case {idx}:\n"
        if kname.startswith("bgemm"):
            kernel1_exec_str += "    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);\n"
        elif kname.startswith("conv2d"):
            kernel1_exec_str += "    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);\n"
        kernel1_exec_str += "    break;\n"
    kernel1_exec_str += "  }\n"

    kernel2_exec_str = ""

    # Get kernel2's execution
    kernel2_exec_str += "  switch (K)\n"
    kernel2_exec_str += "  {\n"
    for idx, kname in enumerate(fusion_sets[1]["Set"]):
        kernel2_exec_str += f"  case {idx}:\n"
        if kname.startswith("bgemm"):
            kernel2_exec_str += "    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);\n"
        elif kname.startswith("conv2d"):
            kernel2_exec_str += "    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);\n"
        kernel2_exec_str += "    break;\n"
    kernel2_exec_str += "  }\n"

    parallel_exec_str = ""

    # Get parallel's execution
    parallel_exec_str += "  switch (K1)\n"
    parallel_exec_str += "  {\n"
    for idx1, kname1 in enumerate(fusion_sets[0]["Set"]):
        parallel_exec_str += f"  case {idx1}:\n"
        parallel_exec_str += "    switch (K2)\n"
        parallel_exec_str += "    {\n"
        for idx2, kname2 in enumerate(fusion_sets[1]["Set"]):
            parallel_exec_str += f"    case {idx2}:\n"
            if kname1.startswith("bgemm"):
                parallel_exec_str += "      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);\n"
            elif kname1.startswith("conv2d"):
                parallel_exec_str += "      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);\n"
            if kname2.startswith("bgemm"):
                parallel_exec_str += "      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);\n"
            elif kname2.startswith("conv2d"):
                parallel_exec_str += "      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);\n"
            parallel_exec_str += "      break;\n"
        parallel_exec_str += "    }\n"
        parallel_exec_str += "    break;\n"
    parallel_exec_str += "  }\n"

    fuse_exec_str = ""

    # Get fuse's execution
    fuse_exec_str += "  switch (K1)\n"
    fuse_exec_str += "  {\n"
    for idx1, kname1 in enumerate(fusion_sets[0]["Set"]):
        fuse_exec_str += f"  case {idx1}:\n"
        fuse_exec_str += "    switch (K2)\n"
        fuse_exec_str += "    {\n"
        for idx2, kname2 in enumerate(fusion_sets[1]["Set"]):
            fuse_exec_str += f"    case {idx2}:\n"
            if kname1.startswith("bgemm"):
                fuse_exec_str += "      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, "
            elif kname1.startswith("conv2d"):
                fuse_exec_str += "      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, "
            if kname2.startswith("bgemm"):
                fuse_exec_str += "I2_gpu, F2_gpu, O2_gpu);\n"
            elif kname2.startswith("conv2d"):
                fuse_exec_str += "O2_gpu, I2_gpu, F2_gpu);\n"
            fuse_exec_str += "      break;\n"
        fuse_exec_str += "    }\n"
        fuse_exec_str += "    break;\n"
    fuse_exec_str += "  }\n"

    return kernel1_exec_str, kernel2_exec_str, parallel_exec_str, fuse_exec_str

#-----------------------------------------------------------------------------------------------
def get_operation_cu(infoYAML):

    kernel1_exec_str, kernel2_exec_str, parallel_exec_str, fuse_exec_str = get_kernel_exec_str(infoYAML)

    operation_cu = \
f"""
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_profiler_api.h>

#include "operation.h"

#define CHECK_CUDA(call)                                              \\
  do                                                                  \\
  {{                                                                   \\
    cudaError_t status_ = call;                                       \\
    if (status_ != cudaSuccess)                                       \\
    {{                                                                 \\
      fprintf(stderr, "CUDA error (%s:%d): %s\\n", __FILE__, __LINE__, \\
              cudaGetErrorString(status_));                           \\
      exit(EXIT_FAILURE);                                             \\
    }}                                                                 \\
  }} while (0)
//----------------------------------------------------------------------------------------------------
float *I1_gpu, *F1_gpu, *O1_gpu;
float *I2_gpu, *F2_gpu, *O2_gpu;

cudaStream_t S1, S2;
//----------------------------------------------------------------------------------------------------
void initialize_kernel1(int *I_shape, int *F_shape, int *O_shape)
{{
  CHECK_CUDA(cudaMalloc(&I1_gpu, I_shape[0] * I_shape[1] * I_shape[2] * I_shape[3] * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F1_gpu, F_shape[0] * F_shape[1] * F_shape[2] * F_shape[3] * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O1_gpu, O_shape[0] * O_shape[1] * O_shape[2] * O_shape[3] * sizeof(float)));

  CHECK_CUDA(cudaStreamCreate(&S1));
}}
//----------------------------------------------------------------------------------------------------
void initialize_kernel2(int *I_shape, int *F_shape, int *O_shape)
{{
  CHECK_CUDA(cudaMalloc(&I2_gpu, I_shape[0] * I_shape[1] * I_shape[2] * I_shape[3] * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F2_gpu, F_shape[0] * F_shape[1] * F_shape[2] * F_shape[3] * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O2_gpu, O_shape[0] * O_shape[1] * O_shape[2] * O_shape[3] * sizeof(float)));

  CHECK_CUDA(cudaStreamCreate(&S2));
}}
//----------------------------------------------------------------------------------------------------
void finalize_kernel1()
{{
  CHECK_CUDA(cudaStreamDestroy(S1));

  CHECK_CUDA(cudaFree(O1_gpu));
  CHECK_CUDA(cudaFree(F1_gpu));
  CHECK_CUDA(cudaFree(I1_gpu));
}}
//----------------------------------------------------------------------------------------------------
void finalize_kernel2()
{{
  CHECK_CUDA(cudaStreamDestroy(S2));

  CHECK_CUDA(cudaFree(O2_gpu));
  CHECK_CUDA(cudaFree(F2_gpu));
  CHECK_CUDA(cudaFree(I2_gpu));
}}
//----------------------------------------------------------------------------------------------------
void run_kernel1(float *I, float *F, float *O, int *I_shape, int *F_shape, int *O_shape,
                 void (*func)(float*, float*, float*), unsigned int *gridDim, unsigned int *blockDim, size_t K)
{{
  // GPU Memory copy (H2D)
  CHECK_CUDA(cudaMemcpy(I1_gpu, I, I_shape[0] * I_shape[1] * I_shape[2] * I_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F, F_shape[0] * F_shape[1] * F_shape[2] * F_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  // Kernel Launch
  dim3 GridDim{{gridDim[0], gridDim[1], gridDim[2]}};
  dim3 BlockDim{{blockDim[0], blockDim[1], blockDim[2]}};
  
{kernel1_exec_str}
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  // GPU Memory copy (D2H)
  CHECK_CUDA(cudaMemcpy(O, O1_gpu, O_shape[0] * O_shape[1] * O_shape[2] * O_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
}}
//----------------------------------------------------------------------------------------------------
void run_kernel2(float *I, float *F, float *O, int *I_shape, int *F_shape, int *O_shape,
                 void (*func)(float*, float*, float*), unsigned int *gridDim, unsigned int *blockDim, size_t K)
{{
  // GPU Memory copy (H2D)
  CHECK_CUDA(cudaMemcpy(I2_gpu, I, I_shape[0] * I_shape[1] * I_shape[2] * I_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F2_gpu, F, F_shape[0] * F_shape[1] * F_shape[2] * F_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  // Kernel Launch
  dim3 GridDim{{gridDim[0], gridDim[1], gridDim[2]}};
  dim3 BlockDim{{blockDim[0], blockDim[1], blockDim[2]}};
  
{kernel2_exec_str}

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  // GPU Memory copy (D2H)
  CHECK_CUDA(cudaMemcpy(O, O2_gpu, O_shape[0] * O_shape[1] * O_shape[2] * O_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
}}
//----------------------------------------------------------------------------------------------------
void run_parallel(float *I1, float *F1, float *O1, float *I2, float *F2, float *O2,
                  int *I1_shape, int *F1_shape, int *O1_shape, int *I2_shape, int *F2_shape, int *O2_shape,
                  void (*func1)(float*, float*, float*), void (*func2)(float*, float*, float*),
                  unsigned int *gridDim1, unsigned int *blockDim1, unsigned int *gridDim2, unsigned int *blockDim2, size_t K1, size_t K2)
{{
  // GPU Memory copy (H2D)
  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, I1_shape[0] * I1_shape[1] * I1_shape[2] * I1_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, F1_shape[0] * F1_shape[1] * F1_shape[2] * F1_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(I2_gpu, I2, I2_shape[0] * I2_shape[1] * I2_shape[2] * I2_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F2_gpu, F2, F2_shape[0] * F2_shape[1] * F2_shape[2] * F2_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  // Kernel Launch
  dim3 GridDim1{{gridDim1[0], gridDim1[1], gridDim1[2]}};
  dim3 BlockDim1{{blockDim1[0], blockDim1[1], blockDim1[2]}};

  dim3 GridDim2{{gridDim2[0], gridDim2[1], gridDim2[2]}};
  dim3 BlockDim2{{blockDim2[0], blockDim2[1], blockDim2[2]}};
  
{parallel_exec_str}
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  // GPU Memory copy (D2H)
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, O1_shape[0] * O1_shape[1] * O1_shape[2] * O1_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O2, O2_gpu, O2_shape[0] * O2_shape[1] * O2_shape[2] * O2_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
}}
//----------------------------------------------------------------------------------------------------
void run_fuse(float *I1, float *F1, float *O1, float *I2, float *F2, float *O2,
              int *I1_shape, int *F1_shape, int *O1_shape, int *I2_shape, int *F2_shape, int *O2_shape,
              void (*func)(float*, float*, float*, float*, float*, float*), unsigned int *gridDim, unsigned int *blockDim, size_t K1, size_t K2)
{{
  // GPU Memory copy (H2D)
  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, I1_shape[0] * I1_shape[1] * I1_shape[2] * I1_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, F1_shape[0] * F1_shape[1] * F1_shape[2] * F1_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(I2_gpu, I2, I2_shape[0] * I2_shape[1] * I2_shape[2] * I2_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F2_gpu, F2, F2_shape[0] * F2_shape[1] * F2_shape[2] * F2_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  // Kernel Launch
  dim3 GridDim{{gridDim[0], gridDim[1], gridDim[2]}};
  dim3 BlockDim{{blockDim[0], blockDim[1], blockDim[2]}};
  
{fuse_exec_str}
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  // GPU Memory copy (D2H)
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, O1_shape[0] * O1_shape[1] * O1_shape[2] * O1_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O2, O2_gpu, O2_shape[0] * O2_shape[1] * O2_shape[2] * O2_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
}}
//----------------------------------------------------------------------------------------------------
"""

    return operation_cu
#-----------------------------------------------------------------------------------------------