#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_profiler_api.h>

#include "operation.h"

#define CHECK_CUDA(call)                                              \
  do                                                                  \
  {                                                                   \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess)                                       \
    {                                                                 \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)
//----------------------------------------------------------------------------------------------------
float *I0_gpu, *F0_gpu, *O0_gpu;
float *I1_gpu, *F1_gpu, *O1_gpu;

cudaStream_t S0, S1;
//----------------------------------------------------------------------------------------------------
/*
 * The kernel functions...
 */

// Original Kernels
extern "C" __global__ void __launch_bounds__(128) bgemm_shared_5120(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT);
extern "C" __global__ void __launch_bounds__(128) bgemm_shared_6144(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT);
extern "C" __global__ void __launch_bounds__(128) bgemm_shared_7168(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT);
extern "C" __global__ void __launch_bounds__(128) bgemm_shared_8192(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT);

extern "C" __global__ void __launch_bounds__(256) conv2d_shared_2048(float* __restrict__ A, float* __restrict__ B, float* __restrict__ W);
extern "C" __global__ void __launch_bounds__(256) conv2d_shared_4096(float* __restrict__ A, float* __restrict__ B, float* __restrict__ W);
extern "C" __global__ void __launch_bounds__(256) conv2d_shared_6144(float* __restrict__ A, float* __restrict__ B, float* __restrict__ W);
extern "C" __global__ void __launch_bounds__(256) conv2d_shared_8192(float* __restrict__ A, float* __restrict__ B, float* __restrict__ W);

extern "C" __global__ void __launch_bounds__(128) softmax_shared_12(float* __restrict__ T_softmax_norm, float* __restrict__ data);
extern "C" __global__ void __launch_bounds__(128) softmax_shared_1036(float* __restrict__ T_softmax_norm, float* __restrict__ data);
extern "C" __global__ void __launch_bounds__(128) softmax_shared_2060(float* __restrict__ T_softmax_norm, float* __restrict__ data);
extern "C" __global__ void __launch_bounds__(128) softmax_shared_3084(float* __restrict__ T_softmax_norm, float* __restrict__ data);
// HFuse
__global__ __launch_bounds__(256, 8) void bgemm_shared_5120_bgemm_shared_5120_copy_fused_kernel_hfuse_lb_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict T_batch_matmul_NT2, float *__restrict A6, float *__restrict B7, float *__restrict T_batch_matmul_NT8);
__global__ __launch_bounds__(256, 8) void bgemm_shared_6144_bgemm_shared_6144_copy_fused_kernel_hfuse_lb_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict T_batch_matmul_NT2, float *__restrict A6, float *__restrict B7, float *__restrict T_batch_matmul_NT8);
// __global__ __launch_bounds__(256, 8) void bgemm_shared_7168_bgemm_shared_7168_copy_fused_kernel_hfuse_lb_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict T_batch_matmul_NT2, float *__restrict A6, float *__restrict B7, float *__restrict T_batch_matmul_NT8);
// __global__ __launch_bounds__(256, 8) void bgemm_shared_8192_bgemm_shared_8192_copy_fused_kernel_hfuse_lb_idx_0(float *__restrict A0, float *__restrict B1, float *__restrict T_batch_matmul_NT2, float *__restrict A6, float *__restrict B7, float *__restrict T_batch_matmul_NT8);

__global__ __launch_bounds__(512, 4) void conv2d_shared_2048_conv2d_shared_2048_copy_fused_kernel_hfuse_lb_idx_1(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10);
__global__ __launch_bounds__(512, 4) void conv2d_shared_4096_conv2d_shared_4096_copy_fused_kernel_hfuse_lb_idx_1(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10);
__global__ __launch_bounds__(512, 4) void conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_hfuse_lb_idx_1(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10);
// __global__ __launch_bounds__(512, 4) void conv2d_shared_8192_conv2d_shared_8192_copy_fused_kernel_hfuse_lb_idx_1(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10);

__global__ __launch_bounds__(256, 8) void softmax_shared_12_softmax_shared_12_copy_fused_kernel_hfuse_lb_idx_0(float *__restrict T_softmax_norm0, float *__restrict data1, float *__restrict T_softmax_norm22, float *__restrict data23);
__global__ __launch_bounds__(256, 8) void softmax_shared_1036_softmax_shared_1036_copy_fused_kernel_hfuse_lb_idx_0(float *__restrict T_softmax_norm0, float *__restrict data1, float *__restrict T_softmax_norm22, float *__restrict data23);
__global__ __launch_bounds__(256, 8) void softmax_shared_2060_softmax_shared_2060_copy_fused_kernel_hfuse_lb_idx_0(float *__restrict T_softmax_norm0, float *__restrict data1, float *__restrict T_softmax_norm22, float *__restrict data23);
__global__ __launch_bounds__(256, 8) void softmax_shared_3084_softmax_shared_3084_copy_fused_kernel_hfuse_lb_idx_0(float *__restrict T_softmax_norm0, float *__restrict data1, float *__restrict T_softmax_norm22, float *__restrict data23);

// BFuse
extern "C" __global__ __launch_bounds__(128) void bgemm_shared_5120_bgemm_shared_5120_copy_fused_kernel_bfuse_idx_0(float *__restrict bgemm_shared_5120_A_, float *__restrict bgemm_shared_5120_B_, float *__restrict bgemm_shared_5120_T_batch_matmul_NT_, float *__restrict bgemm_shared_5120_copy_A_, float *__restrict bgemm_shared_5120_copy_B_, float *__restrict bgemm_shared_5120_copy_T_batch_matmul_NT_);
extern "C" __global__ __launch_bounds__(128) void bgemm_shared_6144_bgemm_shared_6144_copy_fused_kernel_bfuse_idx_0(float *__restrict bgemm_shared_6144_A_, float *__restrict bgemm_shared_6144_B_, float *__restrict bgemm_shared_6144_T_batch_matmul_NT_, float *__restrict bgemm_shared_6144_copy_A_, float *__restrict bgemm_shared_6144_copy_B_, float *__restrict bgemm_shared_6144_copy_T_batch_matmul_NT_);
extern "C" __global__ __launch_bounds__(128) void bgemm_shared_7168_bgemm_shared_7168_copy_fused_kernel_bfuse_idx_0(float *__restrict bgemm_shared_7168_A_, float *__restrict bgemm_shared_7168_B_, float *__restrict bgemm_shared_7168_T_batch_matmul_NT_, float *__restrict bgemm_shared_7168_copy_A_, float *__restrict bgemm_shared_7168_copy_B_, float *__restrict bgemm_shared_7168_copy_T_batch_matmul_NT_);
extern "C" __global__ __launch_bounds__(128) void bgemm_shared_8192_bgemm_shared_8192_copy_fused_kernel_bfuse_idx_0(float *__restrict bgemm_shared_8192_A_, float *__restrict bgemm_shared_8192_B_, float *__restrict bgemm_shared_8192_T_batch_matmul_NT_, float *__restrict bgemm_shared_8192_copy_A_, float *__restrict bgemm_shared_8192_copy_B_, float *__restrict bgemm_shared_8192_copy_T_batch_matmul_NT_);

extern "C" __global__ __launch_bounds__(256) void conv2d_shared_2048_conv2d_shared_2048_copy_fused_kernel_bfuse_idx_0(float *__restrict conv2d_shared_2048_A_, float *__restrict conv2d_shared_2048_B_, float *__restrict conv2d_shared_2048_W_, float *__restrict conv2d_shared_2048_copy_A_, float *__restrict conv2d_shared_2048_copy_B_, float *__restrict conv2d_shared_2048_copy_W_);
extern "C" __global__ __launch_bounds__(256) void conv2d_shared_4096_conv2d_shared_4096_copy_fused_kernel_bfuse_idx_0(float *__restrict conv2d_shared_4096_A_, float *__restrict conv2d_shared_4096_B_, float *__restrict conv2d_shared_4096_W_, float *__restrict conv2d_shared_4096_copy_A_, float *__restrict conv2d_shared_4096_copy_B_, float *__restrict conv2d_shared_4096_copy_W_);
extern "C" __global__ __launch_bounds__(256) void conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_bfuse_idx_0(float *__restrict conv2d_shared_6144_A_, float *__restrict conv2d_shared_6144_B_, float *__restrict conv2d_shared_6144_W_, float *__restrict conv2d_shared_6144_copy_A_, float *__restrict conv2d_shared_6144_copy_B_, float *__restrict conv2d_shared_6144_copy_W_);
extern "C" __global__ __launch_bounds__(256) void conv2d_shared_8192_conv2d_shared_8192_copy_fused_kernel_bfuse_idx_0(float *__restrict conv2d_shared_8192_A_, float *__restrict conv2d_shared_8192_B_, float *__restrict conv2d_shared_8192_W_, float *__restrict conv2d_shared_8192_copy_A_, float *__restrict conv2d_shared_8192_copy_B_, float *__restrict conv2d_shared_8192_copy_W_);

extern "C" __global__ __launch_bounds__(128) void softmax_shared_12_softmax_shared_12_copy_fused_kernel_bfuse_idx_0(float *__restrict softmax_shared_12_T_softmax_norm_, float *__restrict softmax_shared_12_data_, float *__restrict softmax_shared_12_copy_T_softmax_norm_, float *__restrict softmax_shared_12_copy_data_);
extern "C" __global__ __launch_bounds__(128) void softmax_shared_1036_softmax_shared_1036_copy_fused_kernel_bfuse_idx_0(float *__restrict softmax_shared_1036_T_softmax_norm_, float *__restrict softmax_shared_1036_data_, float *__restrict softmax_shared_1036_copy_T_softmax_norm_, float *__restrict softmax_shared_1036_copy_data_);
extern "C" __global__ __launch_bounds__(128) void softmax_shared_2060_softmax_shared_2060_copy_fused_kernel_bfuse_idx_0(float *__restrict softmax_shared_2060_T_softmax_norm_, float *__restrict softmax_shared_2060_data_, float *__restrict softmax_shared_2060_copy_T_softmax_norm_, float *__restrict softmax_shared_2060_copy_data_);
extern "C" __global__ __launch_bounds__(128) void softmax_shared_3084_softmax_shared_3084_copy_fused_kernel_bfuse_idx_0(float *__restrict softmax_shared_3084_T_softmax_norm_, float *__restrict softmax_shared_3084_data_, float *__restrict softmax_shared_3084_copy_T_softmax_norm_, float *__restrict softmax_shared_3084_copy_data_);

// test
extern "C" __global__ __launch_bounds__(256) void bgemm_shared_5120_conv2d_shared_2048_fused_kernel_bfuse_idx_0(float *__restrict bgemm_shared_5120_A_, float *__restrict bgemm_shared_5120_B_, float *__restrict bgemm_shared_5120_T_batch_matmul_NT_, float *__restrict conv2d_shared_2048_A_, float *__restrict conv2d_shared_2048_B_, float *__restrict conv2d_shared_2048_W_);

extern "C" __global__ __launch_bounds__(58) void bgemm_conv2d_fused_kernel_bfuse_idx_0(float *__restrict bgemm_A_, float *__restrict bgemm_B_, float *__restrict bgemm_T_batch_matmul_NT_, float *__restrict conv2d_conv2d_nchw_, float *__restrict conv2d_data_, float *__restrict conv2d_kernel_);
extern "C" __global__ void __launch_bounds__(8) bgemm_test(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT);
extern "C" __global__ void __launch_bounds__(58) conv2d_test(float* __restrict__ conv2d_nchw, float* __restrict__ data, float* __restrict__ kernel);

extern "C" __global__ void __launch_bounds__(256) conv2d_small(float* __restrict__ A, float* __restrict__ B, float* __restrict__ W);
extern "C" __global__ void __launch_bounds__(64) conv2d_large(float* __restrict__ A, float* __restrict__ B, float* __restrict__ W);
__global__ __launch_bounds__(320, 0) void conv2d_small_conv2d_large_fused_kernel_hfuse_idx_1(float *__restrict A0, float *__restrict B1, float *__restrict W2, float *__restrict A8, float *__restrict B9, float *__restrict W10);
extern "C" __global__ __launch_bounds__(256) void conv2d_small_conv2d_large_fused_kernel_bfuse_idx_0(float *__restrict conv2d_small_A_, float *__restrict conv2d_small_B_, float *__restrict conv2d_small_W_, float *__restrict conv2d_large_A_, float *__restrict conv2d_large_B_, float *__restrict conv2d_large_W_);

extern "C" __global__ void __launch_bounds__(232) conv2d_test_2(float* __restrict__ conv2d_nchw, float* __restrict__ data, float* __restrict__ kernel);
extern "C" __global__ void __launch_bounds__(50) softmax_test(float* __restrict__ T_softmax_norm, float* __restrict__ data);
extern "C" __global__ __launch_bounds__(232) void conv2d_softmax_fused_kernel_bfuse_idx_0(float *__restrict conv2d_conv2d_nchw_, float *__restrict conv2d_data_, float *__restrict conv2d_kernel_, float *__restrict softmax_T_softmax_norm_, float *__restrict softmax_data_);

extern "C" __global__ void __launch_bounds__(112) conv2d_test3(float* __restrict__ conv2d_nchw, float* __restrict__ data, float* __restrict__ kernel);
extern "C" __global__ void __launch_bounds__(112) depConv2d(float* __restrict__ DepthwiseConv2d, float* __restrict__ data, float* __restrict__ kernel);
extern "C" __global__ __launch_bounds__(112, 4) void conv2d_depConv2d_fused_kernel_bfuse_idx_0(float *__restrict conv2d_conv2d_nchw_, float *__restrict conv2d_data_, float *__restrict conv2d_kernel_, float *__restrict depConv2d_DepthwiseConv2d_, float *__restrict depConv2d_data_, float *__restrict depConv2d_kernel_);
//----------------------------------------------------------------------------------------------------
void conv2d_initialize()
{
  CHECK_CUDA(cudaMalloc(&I0_gpu, 14 * 14 * 256 * 256 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F0_gpu, 3 * 3 * 256 * 512 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O0_gpu, 14 * 14 * 512 * 256 * sizeof(float)));

  CHECK_CUDA(cudaMalloc(&I1_gpu, 14 * 14 * 256 * 256 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F1_gpu, 3 * 3 * 256 * 512 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O1_gpu, 14 * 14 * 512 * 256 * sizeof(float)));

  CHECK_CUDA(cudaStreamCreate(&S0));
  CHECK_CUDA(cudaStreamCreate(&S1));
  return;
}
//----------------------------------------------------------------------------------------------------
void bgemm_initialize()
{
  CHECK_CUDA(cudaMalloc(&I0_gpu, 128 * 1 * 1024 * 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F0_gpu, 128 * 1 * 1024 * 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O0_gpu, 128 * 1 * 1024 * 1024 * sizeof(float)));

  CHECK_CUDA(cudaMalloc(&I1_gpu, 128 * 1 * 1024 * 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F1_gpu, 128 * 1 * 1024 * 1024 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O1_gpu, 128 * 1 * 1024 * 1024 * sizeof(float)));

  CHECK_CUDA(cudaStreamCreate(&S0));
  CHECK_CUDA(cudaStreamCreate(&S1));
  return;
}
//----------------------------------------------------------------------------------------------------
void softmax_initialize()
{
  CHECK_CUDA(cudaMalloc(&I0_gpu, 128 * 1 * 1 * 1000 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O0_gpu, 128 * 1 * 1 * 1000 * sizeof(float)));

  CHECK_CUDA(cudaMalloc(&I1_gpu, 128 * 1 * 1 * 1000 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O1_gpu, 128 * 1 * 1 * 1000 * sizeof(float)));

  CHECK_CUDA(cudaStreamCreate(&S0));
  CHECK_CUDA(cudaStreamCreate(&S1));
  return;
}
//----------------------------------------------------------------------------------------------------
void test_initialize()
{
  CHECK_CUDA(cudaMalloc(&I0_gpu, 128 * 64 * 56 * 56 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F0_gpu, 128 * 64 * 3 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O0_gpu, 128 * 64 * 56 * 56 * sizeof(float)));

  CHECK_CUDA(cudaMalloc(&I1_gpu, 128 * 128 * 56 * 56 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F1_gpu, 128 * 1 * 3 * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O1_gpu, 128 * 128 * 28 * 28 * sizeof(float)));

  CHECK_CUDA(cudaStreamCreate(&S0));
  CHECK_CUDA(cudaStreamCreate(&S1));
}
//----------------------------------------------------------------------------------------------------
void conv2d_finalize()
{
  CHECK_CUDA(cudaStreamDestroy(S1));
  CHECK_CUDA(cudaStreamDestroy(S0));
  
  CHECK_CUDA(cudaFree(O1_gpu));
  CHECK_CUDA(cudaFree(F1_gpu));
  CHECK_CUDA(cudaFree(I1_gpu));

  CHECK_CUDA(cudaFree(O0_gpu));
  CHECK_CUDA(cudaFree(F0_gpu));
  CHECK_CUDA(cudaFree(I0_gpu));
}
//----------------------------------------------------------------------------------------------------
void bgemm_finalize()
{
  CHECK_CUDA(cudaStreamDestroy(S1));
  CHECK_CUDA(cudaStreamDestroy(S0));
  
  CHECK_CUDA(cudaFree(O1_gpu));
  CHECK_CUDA(cudaFree(F1_gpu));
  CHECK_CUDA(cudaFree(I1_gpu));

  CHECK_CUDA(cudaFree(O0_gpu));
  CHECK_CUDA(cudaFree(F0_gpu));
  CHECK_CUDA(cudaFree(I0_gpu));
}
//----------------------------------------------------------------------------------------------------
void softmax_finalize()
{
  CHECK_CUDA(cudaStreamDestroy(S1));
  CHECK_CUDA(cudaStreamDestroy(S0));
  
  CHECK_CUDA(cudaFree(O1_gpu));
  CHECK_CUDA(cudaFree(I1_gpu));

  CHECK_CUDA(cudaFree(O0_gpu));
  CHECK_CUDA(cudaFree(I0_gpu));
}
//----------------------------------------------------------------------------------------------------
void test_finalize()
{
  CHECK_CUDA(cudaStreamDestroy(S1));
  CHECK_CUDA(cudaStreamDestroy(S0));

  CHECK_CUDA(cudaFree(O1_gpu));
  CHECK_CUDA(cudaFree(F1_gpu));
  CHECK_CUDA(cudaFree(I1_gpu));

  CHECK_CUDA(cudaFree(O0_gpu));
  CHECK_CUDA(cudaFree(F0_gpu));
  CHECK_CUDA(cudaFree(I0_gpu));
}
//----------------------------------------------------------------------------------------------------
void conv2d(float *I, float *F, float *O)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I, 14 * 14 * 256 * 256 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu, F, 3 * 3 * 256 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{1568, 1, 1};
  dim3 blockDim{256, 1, 1};
  conv2d_shared_2048<<<gridDim, blockDim>>>(I0_gpu, O0_gpu, F0_gpu);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O, O0_gpu, 14 * 14 * 512 * 256 * sizeof(float),
                        cudaMemcpyDeviceToHost));
}
//----------------------------------------------------------------------------------------------------
void bgemm(float *I, float *F, float *O)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu, F, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{32768, 1, 1};
  dim3 blockDim{128, 1, 1};
  bgemm_shared_5120<<<gridDim, blockDim>>>(I0_gpu, F0_gpu, O0_gpu);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O, O0_gpu, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyDeviceToHost));
}
//----------------------------------------------------------------------------------------------------
void softmax(float *I, float *O)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{128, 1, 1};
  dim3 blockDim{128, 1, 1};
  softmax_shared_12<<<gridDim, blockDim>>>(O0_gpu, I0_gpu);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O, O0_gpu, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));
}
//----------------------------------------------------------------------------------------------------
void conv2d_parallel(size_t shared_level, float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I0, 14 * 14 * 256 * 256 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu, F0, 3 * 3 * 256 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, 14 * 14 * 256 * 256 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, 3 * 3 * 256 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{1568, 1, 1};
  dim3 blockDim{256, 1, 1};
  switch (shared_level)
  {
  case 0:
    conv2d_shared_2048<<<gridDim, blockDim, 0, S0>>>(I0_gpu, O0_gpu, F0_gpu);
    conv2d_shared_2048<<<gridDim, blockDim, 0, S1>>>(I1_gpu, O1_gpu, F1_gpu);
    break;
  case 1:
    conv2d_shared_4096<<<gridDim, blockDim, 0, S0>>>(I0_gpu, O0_gpu, F0_gpu);
    conv2d_shared_4096<<<gridDim, blockDim, 0, S1>>>(I1_gpu, O1_gpu, F1_gpu);
    break;
  case 2:
    conv2d_shared_6144<<<gridDim, blockDim, 0, S0>>>(I0_gpu, O0_gpu, F0_gpu);
    conv2d_shared_6144<<<gridDim, blockDim, 0, S1>>>(I1_gpu, O1_gpu, F1_gpu);
    break;
  case 3:
    conv2d_shared_8192<<<gridDim, blockDim, 0, S0>>>(I0_gpu, O0_gpu, F0_gpu);
    conv2d_shared_8192<<<gridDim, blockDim, 0, S1>>>(I1_gpu, O1_gpu, F1_gpu);
    break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu, 14 * 14 * 512 * 256 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, 14 * 14 * 512 * 256 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}
//----------------------------------------------------------------------------------------------------
void conv2d_hfuse(size_t shared_level, float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I0, 14 * 14 * 256 * 256 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu, F0, 3 * 3 * 256 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, 14 * 14 * 256 * 256 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, 3 * 3 * 256 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{1568, 1, 1};
  dim3 blockDim{256 + 256, 1, 1};
  switch (shared_level)
  {
  case 0:
    conv2d_shared_2048_conv2d_shared_2048_copy_fused_kernel_hfuse_lb_idx_1<<<gridDim, blockDim>>>(I0_gpu, O0_gpu, F0_gpu, I1_gpu, O1_gpu, F1_gpu);
    break;
  case 1:
    conv2d_shared_4096_conv2d_shared_4096_copy_fused_kernel_hfuse_lb_idx_1<<<gridDim, blockDim>>>(I0_gpu, O0_gpu, F0_gpu, I1_gpu, O1_gpu, F1_gpu);
    break;
  case 2:
    conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_hfuse_lb_idx_1<<<gridDim, blockDim>>>(I0_gpu, O0_gpu, F0_gpu, I1_gpu, O1_gpu, F1_gpu);
    break;
  case 3:
    std::cout << "uses too much shared data\n";
    // conv2d_shared_8192_conv2d_shared_8192_copy_fused_kernel_hfuse_lb_idx_1<<<gridDim, blockDim>>>(I0_gpu, O0_gpu, F0_gpu, I1_gpu, O1_gpu, F1_gpu);
    break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu, 14 * 14 * 512 * 256 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, 14 * 14 * 512 * 256 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}
//----------------------------------------------------------------------------------------------------
void conv2d_bfuse(size_t shared_level, float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I0, 14 * 14 * 256 * 256 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu, F0, 3 * 3 * 256 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, 14 * 14 * 256 * 256 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, 3 * 3 * 256 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{1568 + 1568, 1, 1};
  dim3 blockDim{256, 1, 1};
  switch (shared_level)
  {
  case 0:
    conv2d_shared_2048_conv2d_shared_2048_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu, O0_gpu, F0_gpu, I1_gpu, O1_gpu, F1_gpu);
    break;
  case 1:
    conv2d_shared_4096_conv2d_shared_4096_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu, O0_gpu, F0_gpu, I1_gpu, O1_gpu, F1_gpu);
    break;
  case 2:
    conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu, O0_gpu, F0_gpu, I1_gpu, O1_gpu, F1_gpu);
    break;
  case 3:
    conv2d_shared_8192_conv2d_shared_8192_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu, O0_gpu, F0_gpu, I1_gpu, O1_gpu, F1_gpu);
    break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu, 14 * 14 * 512 * 256 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, 14 * 14 * 512 * 256 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}
//----------------------------------------------------------------------------------------------------
void bgemm_parallel(size_t shared_level, float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I0, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu, F0, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{32768, 1, 1};
  dim3 blockDim{128, 1, 1};
  switch (shared_level)
  {
  case 0:
    bgemm_shared_5120<<<gridDim, blockDim, 0, S0>>>(I0_gpu, F0_gpu, O0_gpu);
    bgemm_shared_5120<<<gridDim, blockDim, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 1:
    bgemm_shared_6144<<<gridDim, blockDim, 0, S0>>>(I0_gpu, F0_gpu, O0_gpu);
    bgemm_shared_6144<<<gridDim, blockDim, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 2:
    bgemm_shared_7168<<<gridDim, blockDim, 0, S0>>>(I0_gpu, F0_gpu, O0_gpu);
    bgemm_shared_7168<<<gridDim, blockDim, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 3:
    bgemm_shared_8192<<<gridDim, blockDim, 0, S0>>>(I0_gpu, F0_gpu, O0_gpu);
    bgemm_shared_8192<<<gridDim, blockDim, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}
//----------------------------------------------------------------------------------------------------
void bgemm_hfuse(size_t shared_level, float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I0, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu, F0, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{32768, 1, 1};
  dim3 blockDim{128 + 128, 1, 1};
  switch (shared_level)
  {
  case 0:
    bgemm_shared_5120_bgemm_shared_5120_copy_fused_kernel_hfuse_lb_idx_0<<<gridDim, blockDim>>>(I0_gpu, F0_gpu, O0_gpu, I1_gpu, F1_gpu, O1_gpu);
    break;
  case 1:
    bgemm_shared_6144_bgemm_shared_6144_copy_fused_kernel_hfuse_lb_idx_0<<<gridDim, blockDim>>>(I0_gpu, F0_gpu, O0_gpu, I1_gpu, F1_gpu, O1_gpu);
    break;
  case 2:
    std::cout << "uses too much shared data\n";
    // bgemm_shared_7168_bgemm_shared_7168_copy_fused_kernel_hfuse_lb_idx_0<<<gridDim, blockDim>>>(I0_gpu, F0_gpu, O0_gpu, I1_gpu, F1_gpu, O1_gpu);
    break;
  case 3:
    std::cout << "uses too much shared data\n";
    // bgemm_shared_8192_bgemm_shared_8192_copy_fused_kernel_hfuse_lb_idx_0<<<gridDim, blockDim>>>(I0_gpu, F0_gpu, O0_gpu, I1_gpu, F1_gpu, O1_gpu);
    break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}
//----------------------------------------------------------------------------------------------------
void bgemm_bfuse(size_t shared_level, float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I0, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu, F0, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{32768 + 32768, 1, 1};
  dim3 blockDim{128, 1, 1};
  switch (shared_level)
  {
  case 0:
    bgemm_shared_5120_bgemm_shared_5120_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu, F0_gpu, O0_gpu, I1_gpu, F1_gpu, O1_gpu);
    break;
  case 1:
    bgemm_shared_6144_bgemm_shared_6144_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu, F0_gpu, O0_gpu, I1_gpu, F1_gpu, O1_gpu);
    break;
  case 2:
    bgemm_shared_7168_bgemm_shared_7168_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu, F0_gpu, O0_gpu, I1_gpu, F1_gpu, O1_gpu);
    break;
  case 3:
    bgemm_shared_8192_bgemm_shared_8192_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu, F0_gpu, O0_gpu, I1_gpu, F1_gpu, O1_gpu);
    break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, 128 * 1 * 1024 * 1024 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}
//----------------------------------------------------------------------------------------------------
void softmax_parallel(size_t shared_level, float *I0, float *O0, float *I1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I0, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{128, 1, 1};
  dim3 blockDim{128, 1, 1};
  switch (shared_level)
  {
  case 0:
    softmax_shared_12<<<gridDim, blockDim, 0, S0>>>(O0_gpu, I0_gpu);
    softmax_shared_12<<<gridDim, blockDim, 0, S1>>>(O1_gpu, I1_gpu);
    break;
  case 1:
    softmax_shared_1036<<<gridDim, blockDim, 0, S0>>>(O0_gpu, I0_gpu);
    softmax_shared_1036<<<gridDim, blockDim, 0, S1>>>(O1_gpu, I1_gpu);
    break;
  case 2:
    softmax_shared_2060<<<gridDim, blockDim, 0, S0>>>(O0_gpu, I0_gpu);
    softmax_shared_2060<<<gridDim, blockDim, 0, S1>>>(O1_gpu, I1_gpu);
    break;
  case 3:
    softmax_shared_3084<<<gridDim, blockDim, 0, S0>>>(O0_gpu, I0_gpu);
    softmax_shared_3084<<<gridDim, blockDim, 0, S1>>>(O1_gpu, I1_gpu);
    break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}
//----------------------------------------------------------------------------------------------------
void softmax_hfuse(size_t shared_level, float *I0, float *O0, float *I1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I0, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{128, 1, 1};
  dim3 blockDim{128 + 128, 1, 1};
  switch (shared_level)
  {
  case 0:
    softmax_shared_12_softmax_shared_12_copy_fused_kernel_hfuse_lb_idx_0<<<gridDim, blockDim>>>(O0_gpu, I0_gpu, O1_gpu, I1_gpu);
    break;
  case 1:
    softmax_shared_1036_softmax_shared_1036_copy_fused_kernel_hfuse_lb_idx_0<<<gridDim, blockDim>>>(O0_gpu, I0_gpu, O1_gpu, I1_gpu);
    break;
  case 2:
    softmax_shared_2060_softmax_shared_2060_copy_fused_kernel_hfuse_lb_idx_0<<<gridDim, blockDim>>>(O0_gpu, I0_gpu, O1_gpu, I1_gpu);
    break;
  case 3:
    softmax_shared_3084_softmax_shared_3084_copy_fused_kernel_hfuse_lb_idx_0<<<gridDim, blockDim>>>(O0_gpu, I0_gpu, O1_gpu, I1_gpu);
    break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}
//----------------------------------------------------------------------------------------------------
void softmax_bfuse(size_t shared_level, float *I0, float *O0, float *I1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I0, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim{128 + 128, 1, 1};
  dim3 blockDim{128, 1, 1};
  switch (shared_level)
  {
  case 0:
    softmax_shared_12_softmax_shared_12_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(O0_gpu, I0_gpu, O1_gpu, I1_gpu);
    break;
  case 1:
    softmax_shared_1036_softmax_shared_1036_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(O0_gpu, I0_gpu, O1_gpu, I1_gpu);
    break;
  case 2:
    softmax_shared_2060_softmax_shared_2060_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(O0_gpu, I0_gpu, O1_gpu, I1_gpu);
    break;
  case 3:
    softmax_shared_3084_softmax_shared_3084_copy_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(O0_gpu, I0_gpu, O1_gpu, I1_gpu);
    break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, 128 * 1 * 1 * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}
//----------------------------------------------------------------------------------------------------
void test(size_t shared_level, float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I0, 32 * 64 * 56 * 56 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu, F0, 32 * 64 * 3 * 3 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, 128 * 128 * 56 * 56 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, 128 * 1 * 3 * 3 * sizeof(float),
                        cudaMemcpyHostToDevice));

  // // parallel
  // dim3 gridDim0{1568, 1, 1}; // conv2d
  // dim3 blockDim0{128, 1, 1};
  // dim3 gridDim1{28672, 1, 1}; // depConv2d
  // dim3 blockDim1{448, 1, 1};

  // conv2d_test3<<<gridDim0, blockDim0, 0, S0>>>(O0_gpu, I0_gpu, F0_gpu);
  // depConv2d<<<gridDim1, blockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
  // CHECK_CUDA(cudaDeviceSynchronize());
  // CHECK_CUDA(cudaGetLastError());

  // BFuse
  dim3 gridDim3{19152, 1, 1};
  dim3 blockDim3{112, 1, 1};

  conv2d_depConv2d_fused_kernel_bfuse_idx_0<<<gridDim3, blockDim3>>>(O0_gpu, I0_gpu, F0_gpu, O1_gpu, I1_gpu, F1_gpu);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu, 32 * 64 * 56 * 56 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, 128 * 128 * 28 * 28 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}
//----------------------------------------------------------------------------------------------------
void test_check(float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu, I0, 32 * 64 * 56 * 56 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu, F0, 32 * 64 * 3 * 3 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, 128 * 128 * 56 * 56 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, 128 * 1 * 3 * 3 * sizeof(float),
                        cudaMemcpyHostToDevice));

  // parallel
  dim3 gridDim0{1, 28, 32}; // conv2d
  dim3 blockDim0{56, 1, 2};
  dim3 gridDim1{1, 1, 16384}; // depConv2d
  dim3 blockDim1{28, 4, 1};

  conv2d_test3<<<gridDim0, blockDim0, 0, S0>>>(O0_gpu, I0_gpu, F0_gpu);
  depConv2d<<<gridDim1, blockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu, 32 * 64 * 56 * 56 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, 128 * 128 * 28 * 28 * sizeof(float),
                        cudaMemcpyDeviceToHost));
}
//----------------------------------------------------------------------------------------------------
void conv2d_bgemm_parallel(size_t shared_level, float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{
  
}
//----------------------------------------------------------------------------------------------------
void conv2d_bgemm_hfuse(size_t shared_level, float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{

}
//----------------------------------------------------------------------------------------------------
void conv2d_bgemm_bfuse(size_t shared_level, float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{

}
//----------------------------------------------------------------------------------------------------