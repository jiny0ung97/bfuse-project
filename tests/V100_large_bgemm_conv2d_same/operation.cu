
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
float *I1_gpu, *F1_gpu, *O1_gpu;
float *I2_gpu, *F2_gpu, *O2_gpu;

cudaStream_t S1, S2;
//----------------------------------------------------------------------------------------------------
void initialize_kernel1(int *I_shape, int *F_shape, int *O_shape)
{
  CHECK_CUDA(cudaMalloc(&I1_gpu, I_shape[0] * I_shape[1] * I_shape[2] * I_shape[3] * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F1_gpu, F_shape[0] * F_shape[1] * F_shape[2] * F_shape[3] * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O1_gpu, O_shape[0] * O_shape[1] * O_shape[2] * O_shape[3] * sizeof(float)));

  CHECK_CUDA(cudaStreamCreate(&S1));
}
//----------------------------------------------------------------------------------------------------
void initialize_kernel2(int *I_shape, int *F_shape, int *O_shape)
{
  CHECK_CUDA(cudaMalloc(&I2_gpu, I_shape[0] * I_shape[1] * I_shape[2] * I_shape[3] * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F2_gpu, F_shape[0] * F_shape[1] * F_shape[2] * F_shape[3] * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O2_gpu, O_shape[0] * O_shape[1] * O_shape[2] * O_shape[3] * sizeof(float)));

  CHECK_CUDA(cudaStreamCreate(&S2));
}
//----------------------------------------------------------------------------------------------------
void finalize_kernel1()
{
  CHECK_CUDA(cudaStreamDestroy(S1));

  CHECK_CUDA(cudaFree(O1_gpu));
  CHECK_CUDA(cudaFree(F1_gpu));
  CHECK_CUDA(cudaFree(I1_gpu));
}
//----------------------------------------------------------------------------------------------------
void finalize_kernel2()
{
  CHECK_CUDA(cudaStreamDestroy(S2));

  CHECK_CUDA(cudaFree(O2_gpu));
  CHECK_CUDA(cudaFree(F2_gpu));
  CHECK_CUDA(cudaFree(I2_gpu));
}
//----------------------------------------------------------------------------------------------------
void run_kernel1(float *I, float *F, float *O, int *I_shape, int *F_shape, int *O_shape,
                 void (*func)(float*, float*, float*), unsigned int *gridDim, unsigned int *blockDim, size_t K)
{
  // GPU Memory copy (H2D)
  CHECK_CUDA(cudaMemcpy(I1_gpu, I, I_shape[0] * I_shape[1] * I_shape[2] * I_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F, F_shape[0] * F_shape[1] * F_shape[2] * F_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  // Kernel Launch
  dim3 GridDim{gridDim[0], gridDim[1], gridDim[2]};
  dim3 BlockDim{blockDim[0], blockDim[1], blockDim[2]};
  
  switch (K)
  {
  case 0:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 1:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 2:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 3:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 4:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 5:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 6:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 7:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 8:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 9:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 10:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 11:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 12:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 13:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 14:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 15:
    func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu);
    break;
  case 16:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 17:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 18:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 19:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 20:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 21:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 22:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 23:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 24:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 25:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 26:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 27:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 28:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  case 29:
    func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu);
    break;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  // GPU Memory copy (D2H)
  CHECK_CUDA(cudaMemcpy(O, O1_gpu, O_shape[0] * O_shape[1] * O_shape[2] * O_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
}
//----------------------------------------------------------------------------------------------------
void run_kernel2(float *I, float *F, float *O, int *I_shape, int *F_shape, int *O_shape,
                 void (*func)(float*, float*, float*), unsigned int *gridDim, unsigned int *blockDim, size_t K)
{
  // GPU Memory copy (H2D)
  CHECK_CUDA(cudaMemcpy(I2_gpu, I, I_shape[0] * I_shape[1] * I_shape[2] * I_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F2_gpu, F, F_shape[0] * F_shape[1] * F_shape[2] * F_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  // Kernel Launch
  dim3 GridDim{gridDim[0], gridDim[1], gridDim[2]};
  dim3 BlockDim{blockDim[0], blockDim[1], blockDim[2]};
  
  switch (K)
  {
  case 0:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 1:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 2:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 3:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 4:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 5:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 6:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 7:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 8:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 9:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 10:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 11:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 12:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 13:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 14:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 15:
    func<<<GridDim, BlockDim>>>(I2_gpu, F2_gpu, O2_gpu);
    break;
  case 16:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 17:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 18:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 19:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 20:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 21:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 22:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 23:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 24:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 25:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 26:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 27:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 28:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  case 29:
    func<<<GridDim, BlockDim>>>(O2_gpu, I2_gpu, F2_gpu);
    break;
  }


  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  // GPU Memory copy (D2H)
  CHECK_CUDA(cudaMemcpy(O, O2_gpu, O_shape[0] * O_shape[1] * O_shape[2] * O_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
}
//----------------------------------------------------------------------------------------------------
void run_parallel(float *I1, float *F1, float *O1, float *I2, float *F2, float *O2,
                  int *I1_shape, int *F1_shape, int *O1_shape, int *I2_shape, int *F2_shape, int *O2_shape,
                  void (*func1)(float*, float*, float*), void (*func2)(float*, float*, float*),
                  unsigned int *gridDim1, unsigned int *blockDim1, unsigned int *gridDim2, unsigned int *blockDim2, size_t K1, size_t K2)
{
  // GPU Memory copy (H2D)
  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, I1_shape[0] * I1_shape[1] * I1_shape[2] * I1_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, F1_shape[0] * F1_shape[1] * F1_shape[2] * F1_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(I2_gpu, I2, I2_shape[0] * I2_shape[1] * I2_shape[2] * I2_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F2_gpu, F2, F2_shape[0] * F2_shape[1] * F2_shape[2] * F2_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  // Kernel Launch
  dim3 GridDim1{gridDim1[0], gridDim1[1], gridDim1[2]};
  dim3 BlockDim1{blockDim1[0], blockDim1[1], blockDim1[2]};

  dim3 GridDim2{gridDim2[0], gridDim2[1], gridDim2[2]};
  dim3 BlockDim2{blockDim2[0], blockDim2[1], blockDim2[2]};
  
  switch (K1)
  {
  case 0:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 1:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 2:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 3:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 4:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 5:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 6:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 7:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 8:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 9:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 10:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 11:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 12:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 13:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 14:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 15:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(I1_gpu, F1_gpu, O1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 16:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 17:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 18:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 19:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 20:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 21:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 22:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 23:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 24:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 25:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 26:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 27:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 28:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 29:
    switch (K2)
    {
    case 0:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func1<<<GridDim1, BlockDim1, 0, S1>>>(O1_gpu, I1_gpu, F1_gpu);
      func2<<<GridDim2, BlockDim2, 0, S2>>>(O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  // GPU Memory copy (D2H)
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, O1_shape[0] * O1_shape[1] * O1_shape[2] * O1_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O2, O2_gpu, O2_shape[0] * O2_shape[1] * O2_shape[2] * O2_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
}
//----------------------------------------------------------------------------------------------------
void run_fuse(float *I1, float *F1, float *O1, float *I2, float *F2, float *O2,
              int *I1_shape, int *F1_shape, int *O1_shape, int *I2_shape, int *F2_shape, int *O2_shape,
              void (*func)(float*, float*, float*, float*, float*, float*), unsigned int *gridDim, unsigned int *blockDim, size_t K1, size_t K2)
{
  // GPU Memory copy (H2D)
  CHECK_CUDA(cudaMemcpy(I1_gpu, I1, I1_shape[0] * I1_shape[1] * I1_shape[2] * I1_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F1_gpu, F1, F1_shape[0] * F1_shape[1] * F1_shape[2] * F1_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(I2_gpu, I2, I2_shape[0] * I2_shape[1] * I2_shape[2] * I2_shape[3] * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F2_gpu, F2, F2_shape[0] * F2_shape[1] * F2_shape[2] * F2_shape[3] * sizeof(float), cudaMemcpyHostToDevice));

  // Kernel Launch
  dim3 GridDim{gridDim[0], gridDim[1], gridDim[2]};
  dim3 BlockDim{blockDim[0], blockDim[1], blockDim[2]};
  
  switch (K1)
  {
  case 0:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 1:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 2:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 3:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 4:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 5:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 6:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 7:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 8:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 9:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 10:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 11:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 12:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 13:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 14:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 15:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(I1_gpu, F1_gpu, O1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 16:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 17:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 18:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 19:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 20:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 21:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 22:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 23:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 24:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 25:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 26:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 27:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 28:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  case 29:
    switch (K2)
    {
    case 0:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 1:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 2:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 3:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 4:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 5:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 6:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 7:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 8:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 9:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 10:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 11:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 12:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 13:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 14:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 15:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, I2_gpu, F2_gpu, O2_gpu);
      break;
    case 16:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 17:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 18:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 19:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 20:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 21:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 22:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 23:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 24:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 25:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 26:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 27:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 28:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    case 29:
      func<<<GridDim, BlockDim>>>(O1_gpu, I1_gpu, F1_gpu, O2_gpu, I2_gpu, F2_gpu);
      break;
    }
    break;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  // GPU Memory copy (D2H)
  CHECK_CUDA(cudaMemcpy(O1, O1_gpu, O1_shape[0] * O1_shape[1] * O1_shape[2] * O1_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(O2, O2_gpu, O2_shape[0] * O2_shape[1] * O2_shape[2] * O2_shape[3] * sizeof(float), cudaMemcpyDeviceToHost));
}
//----------------------------------------------------------------------------------------------------
