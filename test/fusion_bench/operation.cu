#include <cstdio>
#include <cstdlib>
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

float *I0_gpu[2], *F0_gpu[2], *O0_gpu[2];
float *I1_gpu[2], *F1_gpu[2], *O1_gpu[2];

cudaStream_t streams[2];
static size_t B[2] = {1, 1};

/**************************************************************************/

extern "C" __global__ void __launch_bounds__(112) conv2d_B1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nhwc);
extern "C" __global__ void __launch_bounds__(112) conv2d_B2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nhwc);
extern "C" __global__ void __launch_bounds__(128) conv2d_B4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nhwc);
extern "C" __global__ void __launch_bounds__(128) conv2d_B8(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nhwc);

extern "C" __global__ void __launch_bounds__(50) matmul_B16(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ T_matmul_NT);
extern "C" __global__ void __launch_bounds__(32) matmul_B32(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ T_matmul_NT);
extern "C" __global__ void __launch_bounds__(32) matmul_B64(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ T_matmul_NT);
extern "C" __global__ void __launch_bounds__(32) matmul_B128(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ T_matmul_NT);

/**************************************************************************/

// extern "C" __global__ __launch_bounds__(112, 0) void conv2d_B1_matmul_B16_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
// extern "C" __global__ __launch_bounds__(112, 0) void conv2d_B1_matmul_B32_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
// extern "C" __global__ __launch_bounds__(112, 0) void conv2d_B1_matmul_B64_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
// extern "C" __global__ __launch_bounds__(112, 0) void conv2d_B1_matmul_B128_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);

// extern "C" __global__ __launch_bounds__(112, 0) void conv2d_B2_matmul_B16_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
// extern "C" __global__ __launch_bounds__(112, 0) void conv2d_B2_matmul_B32_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
// extern "C" __global__ __launch_bounds__(112, 0) void conv2d_B2_matmul_B64_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
// extern "C" __global__ __launch_bounds__(112, 0) void conv2d_B2_matmul_B128_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);

// extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B4_matmul_B16_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B4_matmul_B32_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B4_matmul_B64_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B4_matmul_B128_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);

// extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B8_matmul_B16_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B8_matmul_B32_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B8_matmul_B64_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B8_matmul_B128_fused_kernel_hfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);

/**************************************************************************/

extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B4_matmul_B32_fused_kernel_bfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B4_matmul_B64_fused_kernel_bfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B4_matmul_B128_fused_kernel_bfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);

extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B8_matmul_B32_fused_kernel_bfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B8_matmul_B64_fused_kernel_bfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);
extern "C" __global__ __launch_bounds__(128, 0) void conv2d_B8_matmul_B128_fused_kernel_bfuse_idx_0(float *__restrict data0, float *__restrict kernel1, float *__restrict conv2d_nhwc2, float *__restrict data6, float *__restrict weight7, float *__restrict T_matmul_NT8);

/**************************************************************************/

extern "C" __global__ __launch_bounds__(128) void conv2d_B4_matmul_B32_fused_(float *__restrict conv2d_B4_data_, float *__restrict conv2d_B4_kernel_, float *__restrict conv2d_B4_conv2d_nhwc_, float *__restrict matmul_B32_data_, float *__restrict matmul_B32_weight_, float *__restrict matmul_B32_T_matmul_NT_);
extern "C" __global__ __launch_bounds__(128) void conv2d_B4_matmul_B64_fused_(float *__restrict conv2d_B4_data_, float *__restrict conv2d_B4_kernel_, float *__restrict conv2d_B4_conv2d_nhwc_, float *__restrict matmul_B64_data_, float *__restrict matmul_B64_weight_, float *__restrict matmul_B64_T_matmul_NT_);
extern "C" __global__ __launch_bounds__(128) void conv2d_B4_matmul_B128_fused_(float *__restrict conv2d_B4_data_, float *__restrict conv2d_B4_kernel_, float *__restrict conv2d_B4_conv2d_nhwc_, float *__restrict matmul_B128_data_, float *__restrict matmul_B128_weight_, float *__restrict matmul_B128_T_matmul_NT_);

extern "C" __global__ __launch_bounds__(128) void conv2d_B8_matmul_B32_fused_(float *__restrict conv2d_B8_data_, float *__restrict conv2d_B8_kernel_, float *__restrict conv2d_B8_conv2d_nhwc_, float *__restrict matmul_B32_data_, float *__restrict matmul_B32_weight_, float *__restrict matmul_B32_T_matmul_NT_);
extern "C" __global__ __launch_bounds__(128) void conv2d_B8_matmul_B64_fused_(float *__restrict conv2d_B8_data_, float *__restrict conv2d_B8_kernel_, float *__restrict conv2d_B8_conv2d_nhwc_, float *__restrict matmul_B64_data_, float *__restrict matmul_B64_weight_, float *__restrict matmul_B64_T_matmul_NT_);
extern "C" __global__ __launch_bounds__(128) void conv2d_B8_matmul_B128_fused_(float *__restrict conv2d_B8_data_, float *__restrict conv2d_B8_kernel_, float *__restrict conv2d_B8_conv2d_nhwc_, float *__restrict matmul_B128_data_, float *__restrict matmul_B128_weight_, float *__restrict matmul_B128_T_matmul_NT_);

/**************************************************************************/

void initialize(size_t b[2])
{
  for (int i = 0; i < 2; ++i) {
    CHECK_CUDA(cudaMalloc(&I0_gpu[i], b[i] * 56 * 56 * 64 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&F0_gpu[i], 3 * 3 * 64 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&O0_gpu[i], b[i] * 28 * 28 * 128 * sizeof(float)));
  }

  for (int i = 0; i < 2; ++i) {
    CHECK_CUDA(cudaMalloc(&I1_gpu[i], b[1 - i] * 512 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&F1_gpu[i], 1000 * 512 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&O1_gpu[i], b[1 - i] * 1000 * sizeof(float)));
  }

  CHECK_CUDA(cudaStreamCreate(&streams[0]));
  CHECK_CUDA(cudaStreamCreate(&streams[1]));

  B[0] = b[0]; B[1] = b[1];
  return;
}

void finalize()
{
  for (int i = 0; i < 2; ++i) {
    CHECK_CUDA(cudaFree(I0_gpu[i]));
    CHECK_CUDA(cudaFree(F0_gpu[i]));
    CHECK_CUDA(cudaFree(O0_gpu[i]));
  }

  for (int i = 0; i < 2; ++i) {
    CHECK_CUDA(cudaFree(I1_gpu[i]));
    CHECK_CUDA(cudaFree(F1_gpu[i]));
    CHECK_CUDA(cudaFree(O1_gpu[i]));
  }

  CHECK_CUDA(cudaStreamDestroy(streams[0]));
  CHECK_CUDA(cudaStreamDestroy(streams[1]));
  return;
}

void conv2d(float *I, float *F, float *O)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu[0], I, B[0] * 56 * 56 * 64 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu[0], F, 3 * 3 * 64 * 128 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim;
  dim3 blockDim;
  switch(B[0])
  {
    case 1:
      gridDim  = {224, 1, 1};
      blockDim = {112, 1, 1};
      conv2d_B1<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
      break;
    case 2:
      gridDim  = {224, 1, 1};
      blockDim = {112, 1, 1};
      conv2d_B2<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
      break;
    case 4:
      gridDim  = {224, 1, 1};
      blockDim = {128, 1, 1};
      conv2d_B4<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
      break;
    case 8:
      gridDim  = {392, 1, 1};
      blockDim = {128, 1, 1};
      conv2d_B8<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
      break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O, O0_gpu[0], B[0] * 28 * 28 * 128 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}

void matmul(float *I, float *F, float *O)
{
  CHECK_CUDA(cudaMemcpy(I1_gpu[0], I, B[1] * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu[0], F, 1000 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim;
  dim3 blockDim;
  switch(B[1])
  {
    case 16:
      gridDim  = {80, 1, 1};
      blockDim = {50, 1, 1};
      matmul_B16<<<gridDim, blockDim>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
      break;
    case 32:
      gridDim  = {125, 1, 1};
      blockDim = {32, 1, 1};
      matmul_B32<<<gridDim, blockDim>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
      break;
    case 64:
      gridDim  = {400, 1, 1};
      blockDim = {32, 1, 1};
      matmul_B64<<<gridDim, blockDim>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
      break;
    case 128:
      gridDim  = {500, 1, 1};
      blockDim = {32, 1, 1};
      matmul_B128<<<gridDim, blockDim>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
      break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O, O1_gpu[0], B[1] * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}

void conv2d_matmul_parallel(float *I0, float *F0, float *O0,
                            float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu[0], I0, B[0] * 56 * 56 * 64 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu[0], F0, 3 * 3 * 64 * 128 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(I1_gpu[0], I1, B[1] * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu[0], F1, 1000 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim0, gridDim1;
  dim3 blockDim0, blockDim1;
  switch(B[0])
  {
    case 1:
      gridDim0  = {224, 1, 1};
      blockDim0 = {112, 1, 1};
      switch(B[1])
      {
        case 16:
          gridDim1  = {80, 1, 1};
          blockDim1 = {50, 1, 1};
          matmul_B16<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B1<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 32:
          gridDim1  = {125, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B32<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B1<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 64:
          gridDim1  = {400, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B64<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B1<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 128:
          gridDim1  = {500, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B128<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B1<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
      }
      break;
    case 2:
      gridDim0  = {224, 1, 1};
      blockDim0 = {112, 1, 1};
      switch(B[1])
      {
        case 16:
          gridDim1  = {80, 1, 1};
          blockDim1 = {50, 1, 1};
          matmul_B16<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B2<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 32:
          gridDim1  = {125, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B32<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B2<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 64:
          gridDim1  = {400, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B64<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B2<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 128:
          gridDim1  = {500, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B128<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B2<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
      }
      break;
    case 4:
      gridDim0  = {224, 1, 1};
      blockDim0 = {128, 1, 1};
      switch(B[1])
      {
        case 16:
          gridDim1  = {80, 1, 1};
          blockDim1 = {50, 1, 1};
          matmul_B16<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B4<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 32:
          gridDim1  = {125, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B32<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B4<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 64:
          gridDim1  = {400, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B64<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B4<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 128:
          gridDim1  = {500, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B128<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B4<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
      }
      break;
    case 8:
      gridDim0  = {392, 1, 1};
      blockDim0 = {128, 1, 1};
      switch(B[1])
      {
        case 16:
          gridDim1  = {80, 1, 1};
          blockDim1 = {50, 1, 1};
          matmul_B16<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B8<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 32:
          gridDim1  = {125, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B32<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B8<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 64:
          gridDim1  = {400, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B64<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B8<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
        case 128:
          gridDim1  = {500, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B128<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          conv2d_B8<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          break;
      }
      break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu[0], B[0] * 28 * 28 * 128 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu[0], B[1] * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));

  return;
}

void conv2d_matmul_fuse(size_t type, float *I0, float *F0, float *O0,
                        float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu[0], I0, B[0] * 56 * 56 * 64 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu[0], F0, 3 * 3 * 64 * 128 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(I1_gpu[0], I1, B[1] * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu[0], F1, 1000 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim;
  dim3 blockDim;
  switch(B[0])
  {
    // case 1:
    //   switch(B[1])
    //   {
    //     case 16:
    //       // gridDim  = {224 + 80, 1, 1};
    //       // blockDim = {112}; // 112 vs 50
    //       gridDim  = {224, 1, 1}; // 224 vs 80
    //       blockDim = {112 + 50};
    //       conv2d_B1_matmul_B16_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
    //       break;
    //     case 32:
    //       // gridDim  = {224 + 125, 1, 1};
    //       // blockDim = {112}; // 112 vs 32
    //       gridDim  = {224, 1, 1}; // 224 vs 125
    //       blockDim = {112 + 32};
    //       conv2d_B1_matmul_B32_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
    //       break;
    //     case 64:
    //       // gridDim  = {224 + 400, 1, 1};
    //       // blockDim = {112}; // 112 vs 32
    //       gridDim  = {400, 1, 1}; // 224 vs 400
    //       blockDim = {112 + 32};
    //       conv2d_B1_matmul_B64_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
    //       break;
    //     case 128:
    //       // gridDim  = {224 + 500, 1, 1};
    //       // blockDim = {112}; // 112 vs 32
    //       gridDim  = {500, 1, 1}; // 224 vs 500
    //       blockDim = {112 + 32};
    //       conv2d_B1_matmul_B128_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
    //       break;
    //   }
    //   break;
    // case 2:
    //   switch(B[1])
    //   {
    //     case 16:
    //       // gridDim  = {224 + 80, 1, 1};
    //       // blockDim = {112}; // 112 vs 50
    //       gridDim  = {224, 1, 1}; // 224 vs 80
    //       blockDim = {112 + 50};
    //       conv2d_B2_matmul_B16_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
    //       break;
    //     case 32:
    //       // gridDim  = {224 + 125, 1, 1};
    //       // blockDim = {112}; // 112 vs 32
    //       gridDim  = {224, 1, 1}; // 224 vs 125
    //       blockDim = {112 + 32};
    //       conv2d_B2_matmul_B32_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
    //       break;
    //     case 64:
    //       // gridDim  = {224 + 400, 1, 1};
    //       // blockDim = {112}; // 112 vs 32
    //       gridDim  = {400, 1, 1}; // 224 vs 400
    //       blockDim = {112 + 32};
    //       conv2d_B2_matmul_B64_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
    //       break;
    //     case 128:
    //       // gridDim  = {224 + 500, 1, 1};
    //       // blockDim = {112}; // 112 vs 32
    //       gridDim  = {500, 1, 1}; // 224 vs 500
    //       blockDim = {112 + 32};
    //       conv2d_B2_matmul_B128_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
    //       break;
    //   }
    //   break;
    case 4:
      switch(B[1])
      {
        // case 16:
        //   // gridDim  = {224 + 80, 1, 1};
        //   // blockDim = {128}; // 128 vs 50
        //   gridDim  = {224, 1, 1}; // 224 vs 80
        //   blockDim = {128 + 50};
        //   conv2d_B4_matmul_B16_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
        //   break;
        case 32:
          if (type == 1) {
            gridDim  = {224, 1, 1}; // 224 vs 125
            blockDim = {128 + 32};
            conv2d_B4_matmul_B32_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          }
          else if (type == 2) {
            gridDim  = {224 + 125, 1, 1};
            blockDim = {128}; // 128 vs 32
            conv2d_B4_matmul_B32_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          }
          else if (type == 3) {
            gridDim  = {224 + 125, 1, 1};
            blockDim = {128}; // 128 vs 32
            conv2d_B4_matmul_B32_fused_<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);  
          }
          break;
        case 64:
          if (type == 1) {
            gridDim  = {400, 1, 1}; // 224 vs 400
            blockDim = {128 + 32};
            conv2d_B4_matmul_B64_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          } else if (type == 2) {
            gridDim  = {224 + 400, 1, 1};
            blockDim = {128}; // 128 vs 32
            conv2d_B4_matmul_B64_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          } else if (type == 3) {
            gridDim  = {224 + 400, 1, 1};
            blockDim = {128}; // 128 vs 32
            conv2d_B4_matmul_B64_fused_<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          }
          break;
        case 128:
          if (type == 1) {
            gridDim  = {500, 1, 1}; // 224 vs 500
            blockDim = {128 + 32};
            conv2d_B4_matmul_B128_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          } else if (type == 2) {
            gridDim  = {224 + 500, 1, 1};
            blockDim = {128}; // 128 vs 32
            conv2d_B4_matmul_B128_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          } else if (type == 3) {
            gridDim  = {224 + 500, 1, 1};
            blockDim = {128}; // 128 vs 32
            conv2d_B4_matmul_B128_fused_<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          }
          break;
      }
      break;
    case 8:
      switch(B[1])
      {
        // case 16:
        //   // gridDim  = {392 + 80, 1, 1};
        //   // blockDim = {128}; // 128 vs 50
        //   gridDim  = {392, 1, 1}; // 392 vs 80
        //   blockDim = {128 + 50};
        //   conv2d_B8_matmul_B16_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
        //   break;
        case 32:
          if (type == 1) {
            gridDim  = {392, 1, 1}; // 392 vs 125
            blockDim = {128 + 32};
            conv2d_B8_matmul_B32_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          } else if (type == 2) {
            gridDim  = {392 + 125, 1, 1};
            blockDim = {128}; // 128 vs 32
            conv2d_B8_matmul_B32_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          } else if (type == 3) {
            gridDim  = {392 + 125, 1, 1};
            blockDim = {128}; // 128 vs 32
            conv2d_B8_matmul_B32_fused_<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          }
          break;
        case 64:
          if (type == 1) {
          gridDim  = {400, 1, 1}; // 392 vs 400
          blockDim = {128 + 32};
            conv2d_B8_matmul_B64_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          } else if (type == 2) {
          gridDim  = {392 + 400, 1, 1};
          blockDim = {128}; // 128 vs 32
          conv2d_B8_matmul_B32_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          } else if (type == 3) {
          gridDim  = {392 + 400, 1, 1};
          blockDim = {128}; // 128 vs 32
            conv2d_B8_matmul_B64_fused_<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          }
          break;
        case 128:
          if (type == 1) {
            gridDim  = {500, 1, 1}; // 392 vs 500
            blockDim = {128 + 32};
            conv2d_B8_matmul_B128_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          } else if (type == 2) {
            gridDim  = {392 + 500, 1, 1};
            blockDim = {128}; // 128 vs 32
            conv2d_B8_matmul_B32_fused_kernel_bfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          } else if (type == 3) {
            gridDim  = {392 + 500, 1, 1};
            blockDim = {128}; // 128 vs 32
            conv2d_B8_matmul_B128_fused_<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          }
          break;
      }
      break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu[0], B[0] * 28 * 28 * 128 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu[0], B[1] * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));

  return;
}

void conv2d_conv2d_parallel(float *I0, float *F0, float *O0,
                            float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I0_gpu[0], I0, B[0] * 56 * 56 * 64 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu[0], F0, 3 * 3 * 64 * 128 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(I0_gpu[1], I1, B[0] * 56 * 56 * 64 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F0_gpu[1], F1, 3 * 3 * 64 * 128 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim0, gridDim1;
  dim3 blockDim0, blockDim1;
  switch(B[0])
  {
    case 1:
      gridDim0  = {224, 1, 1};
      blockDim0 = {112, 1, 1};
      switch(B[1])
      {
        case 2:
          gridDim1  = {224, 1, 1};
          blockDim1 = {112, 1, 1};
          conv2d_B1<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          conv2d_B2<<<gridDim1, blockDim1, 0, streams[1]>>>(I0_gpu[1], F0_gpu[1], O0_gpu[1]);
          break;
        case 4:
          gridDim1  = {224, 1, 1};
          blockDim1 = {128, 1, 1};
          conv2d_B1<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          conv2d_B4<<<gridDim1, blockDim1, 0, streams[1]>>>(I0_gpu[1], F0_gpu[1], O0_gpu[1]);
          break;
        case 8:
          gridDim1  = {392, 1, 1};
          blockDim1 = {128, 1, 1};
          conv2d_B1<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          conv2d_B8<<<gridDim1, blockDim1, 0, streams[1]>>>(I0_gpu[1], F0_gpu[1], O0_gpu[1]);
          break;
      }
      break;
    case 2:
      gridDim0  = {224, 1, 1};
      blockDim0 = {112, 1, 1};
      switch(B[1])
      {
        case 4:
          gridDim1  = {224, 1, 1};
          blockDim1 = {128, 1, 1};
          conv2d_B2<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          conv2d_B4<<<gridDim1, blockDim1, 0, streams[1]>>>(I0_gpu[1], F0_gpu[1], O0_gpu[1]);
          break;
        case 8:
          gridDim1  = {392, 1, 1};
          blockDim1 = {128, 1, 1};
          conv2d_B2<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          conv2d_B8<<<gridDim1, blockDim1, 0, streams[1]>>>(I0_gpu[1], F0_gpu[1], O0_gpu[1]);
          break;
      }
      break;
    case 4:
      gridDim0  = {224, 1, 1};
      blockDim0 = {128, 1, 1};
      switch(B[1])
      {
        case 8:
          gridDim1  = {392, 1, 1};
          blockDim1 = {128, 1, 1};
          conv2d_B4<<<gridDim0, blockDim0, 0, streams[0]>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0]);
          conv2d_B8<<<gridDim1, blockDim1, 0, streams[1]>>>(I0_gpu[1], F0_gpu[1], O0_gpu[1]);
          break;
      }
      break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O0_gpu[0], B[0] * 28 * 28 * 128 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O0_gpu[1], B[0] * 28 * 28 * 128 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}

void conv2d_conv2d_BFuse(float *I0, float *F0, float *O0,
                         float *I1, float *F1, float *O1)
{
  // CHECK_CUDA(cudaMemcpy(I0_gpu[0], I0, B[0] * 56 * 56 * 64 * sizeof(float),
  //                       cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(F0_gpu[0], F0, 3 * 3 * 64 * 128 * sizeof(float),
  //                       cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(I0_gpu[1], I1, B[0] * 56 * 56 * 64 * sizeof(float),
  //                       cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(F0_gpu[1], F1, 3 * 3 * 64 * 128 * sizeof(float),
  //                       cudaMemcpyHostToDevice));

  // dim3 gridDim;
  // dim3 blockDim;
  // switch(B[0])
  // {
  //   case 1:
  //     switch(B[1])
  //     {
  //       case 2:
  //         gridDim  = {224 + 224, 1, 1};
  //         blockDim = {112, 1, 1}; // 112 vs 112
  //         conv2d_B1_conv2d_B2_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I0_gpu[1], F0_gpu[1], O0_gpu[1]);
  //         break;
  //       case 4:
  //         gridDim  = {224 + 224, 1, 1};
  //         blockDim = {128, 1, 1}; // 112 vs 128
  //         conv2d_B1_conv2d_B4_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I0_gpu[1], F0_gpu[1], O0_gpu[1]);
  //         break;
  //       case 8:
  //         gridDim  = {224 + 392, 1, 1};
  //         blockDim = {128, 1, 1}; // 112 vs 128
  //         conv2d_B1_conv2d_B8_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I0_gpu[1], F0_gpu[1], O0_gpu[1]);
  //         break;
  //     }
  //     break;
  //   case 2:
  //     switch(B[1])
  //     {
  //       case 4:
  //         gridDim  = {224 + 224, 1, 1};
  //         blockDim = {128, 1, 1}; // 112 vs 128
  //         conv2d_B2_conv2d_B4_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I0_gpu[1], F0_gpu[1], O0_gpu[1]);
  //         break;
  //       case 8:
  //         gridDim  = {224 + 392, 1, 1};
  //         blockDim = {128, 1, 1}; // 112 vs 128
  //         conv2d_B2_conv2d_B8_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I0_gpu[1], F0_gpu[1], O0_gpu[1]);
  //         break;
  //     }
  //     break;
  //   case 4:
  //     switch(B[1])
  //     {
  //       case 8:
  //         gridDim  = {224 + 392, 1, 1};
  //         blockDim = {128, 1, 1}; // 128 vs 128
  //         conv2d_B4_conv2d_B8_fused_kernel_hfuse_idx_0<<<gridDim, blockDim>>>(I0_gpu[0], F0_gpu[0], O0_gpu[0], I0_gpu[1], F0_gpu[1], O0_gpu[1]);
  //         break;
  //     }
  //     break;
  // }
  // CHECK_CUDA(cudaDeviceSynchronize());
  // CHECK_CUDA(cudaGetLastError());

  // CHECK_CUDA(cudaMemcpy(O0, O0_gpu[0], B[0] * 28 * 28 * 128 * sizeof(float),
  //                       cudaMemcpyDeviceToHost));
  // CHECK_CUDA(cudaMemcpy(O1, O0_gpu[1], B[0] * 28 * 28 * 128 * sizeof(float),
  //                       cudaMemcpyDeviceToHost));
  return;
}

void matmul_matmul_parallel(float *I0, float *F0, float *O0,
                            float *I1, float *F1, float *O1)
{
  CHECK_CUDA(cudaMemcpy(I1_gpu[0], I0, B[1] * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu[0], F0, 1000 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(I1_gpu[1], I1, B[1] * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu[1], F1, 1000 * 512 * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 gridDim0, gridDim1;
  dim3 blockDim0, blockDim1;
  switch(B[1])
  {
    case 16:
      gridDim0  = {80, 1, 1};
      blockDim0 = {50, 1, 1};
      switch(B[0])
      {
        case 32:
          gridDim1  = {125, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B16<<<gridDim0, blockDim0, 0, streams[0]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          matmul_B32<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[1], F1_gpu[1], O1_gpu[1]);
          break;
        case 64:
          gridDim1  = {400, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B16<<<gridDim0, blockDim0, 0, streams[0]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          matmul_B64<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[1], F1_gpu[1], O1_gpu[1]);
          break;
        case 128:
          gridDim1  = {500, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B16<<<gridDim0, blockDim0, 0, streams[0]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          matmul_B128<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[1], F1_gpu[1], O1_gpu[1]);
          break;
      }
      break;
    case 32:
      gridDim0  = {125, 1, 1};
      blockDim0 = {32, 1, 1};
      switch(B[0])
      {
        case 64:
          gridDim1  = {400, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B32<<<gridDim0, blockDim0, 0, streams[0]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          matmul_B64<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[1], F1_gpu[1], O1_gpu[1]);
          break;
        case 128:
          gridDim1  = {500, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B32<<<gridDim0, blockDim0, 0, streams[0]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          matmul_B128<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[1], F1_gpu[1], O1_gpu[1]);
          break;
      }
      break;
    case 64:
      gridDim0  = {400, 1, 1};
      blockDim0 = {32, 1, 1};
      switch(B[0])
      {
        case 128:
          gridDim1  = {500, 1, 1};
          blockDim1 = {32, 1, 1};
          matmul_B16<<<gridDim0, blockDim0, 0, streams[0]>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0]);
          matmul_B128<<<gridDim1, blockDim1, 0, streams[1]>>>(I1_gpu[1], F1_gpu[1], O1_gpu[1]);
          break;
      }
      break;
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O0, O1_gpu[0], B[1] * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu[1], B[1] * 1000 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return;
}

void matmul_matmul_BFuse(float *I0, float *F0, float *O0,
                         float *I1, float *F1, float *O1)
{
  // CHECK_CUDA(cudaMemcpy(I1_gpu[0], I0, B[1] * 512 * sizeof(float),
  //                       cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(F1_gpu[0], F0, 1000 * 512 * sizeof(float),
  //                       cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(I1_gpu[1], I1, B[1] * 512 * sizeof(float),
  //                       cudaMemcpyHostToDevice));
  // CHECK_CUDA(cudaMemcpy(F1_gpu[1], F1, 1000 * 512 * sizeof(float),
  //                       cudaMemcpyHostToDevice));

  // dim3 gridDim;
  // dim3 blockDim;
  // switch(B[1])
  // {
  //   case 16:
  //     switch(B[0])
  //     {
  //       case 32:
  //         gridDim  = {80 + 125, 1, 1};
  //         blockDim = {50, 1, 1}; // 50 vs 32
  //         matmul_B16_matmul_B32_fused_kernel_vfuse_idx_0<<<gridDim, blockDim>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0], I1_gpu[1], F1_gpu[1], O1_gpu[1]);
  //         break;
  //       case 64:
  //         gridDim  = {80 + 400, 1, 1};
  //         blockDim = {50, 1, 1}; // 50 vs 32
  //         matmul_B16_matmul_B64_fused_kernel_vfuse_idx_0<<<gridDim, blockDim>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0], I1_gpu[1], F1_gpu[1], O1_gpu[1]);
  //         break;
  //       case 128:
  //         gridDim  = {80 + 500, 1, 1};
  //         blockDim = {50, 1, 1}; // 50 vs 32
  //         matmul_B16_matmul_B128_fused_kernel_vfuse_idx_0<<<gridDim, blockDim>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0], I1_gpu[1], F1_gpu[1], O1_gpu[1]);
  //         break;
  //     }
  //     break;
  //   case 32:
  //     switch(B[0])
  //     {
  //       case 64:
  //         gridDim  = {125 + 400, 1, 1};
  //         blockDim = {32, 1, 1}; // 32 vs 32
  //         matmul_B32_matmul_B64_fused_kernel_vfuse_idx_0<<<gridDim, blockDim>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0], I1_gpu[1], F1_gpu[1], O1_gpu[1]);
  //         break;
  //       case 128:
  //         gridDim  = {125 + 500, 1, 1};
  //         blockDim = {32, 1, 1}; // 32 vs 32
  //         matmul_B32_matmul_B128_fused_kernel_vfuse_idx_0<<<gridDim, blockDim>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0], I1_gpu[1], F1_gpu[1], O1_gpu[1]);
  //         break;
  //     }
  //     break;
  //   case 64:
  //     switch(B[0])
  //     {
  //       case 128:
  //         gridDim  = {400 + 500, 1, 1};
  //         blockDim = {32, 1, 1}; // 32 vs 32
  //         matmul_B64_matmul_B128_fused_kernel_vfuse_idx_0<<<gridDim, blockDim>>>(I1_gpu[0], F1_gpu[0], O1_gpu[0], I1_gpu[1], F1_gpu[1], O1_gpu[1]);
  //         break;
  //     }
  //     break;
  // }
  // CHECK_CUDA(cudaDeviceSynchronize());
  // CHECK_CUDA(cudaGetLastError());

  // CHECK_CUDA(cudaMemcpy(O0, O1_gpu[0], B[1] * 1000 * sizeof(float),
  //                       cudaMemcpyDeviceToHost));
  // CHECK_CUDA(cudaMemcpy(O1, O1_gpu[1], B[1] * 1000 * sizeof(float),
  //                       cudaMemcpyDeviceToHost));
  return;
}