
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif


#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(128) softmax_shared_2060(float* __restrict__ T_softmax_norm, float* __restrict__ data) {
  float normal_reduce_temp0[1];
  __shared__ float red_result[1+512];//
  __shared__ float T_softmax_maxelem[1+512];//
  float normal_reduce_temp0_1[1];
  __shared__ float red_result_1[1+512];//
  __shared__ float T_softmax_expsum[1+512];//
  normal_reduce_temp0[0] = -3.402823e+38f;
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[((((int)blockIdx.x) * 1000) + ((int)threadIdx.x))]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 128)]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 256)]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 384)]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 512)]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 640)]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 768)]);
  if (((int)threadIdx.x) < 104) {
    normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 896)]);
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[4];
  red_buf0_1[0] = normal_reduce_temp0[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int)threadIdx.x) % 32) == 0) {
    red_buf_staging[(((int)threadIdx.x) >> 5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    red_buf0[0] = red_buf_staging[((int)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)15);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int)threadIdx.x) == 0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_softmax_maxelem[0] = ((volatile float*)red_result)[0];
  }
  normal_reduce_temp0_1[0] = 0.000000e+00f;
  __syncthreads();
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    if (((k_outer * 16) + (((int)threadIdx.x) >> 3)) < 125) {
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + (k_outer * 128)) + ((int)threadIdx.x))] - T_softmax_maxelem[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[4];
  red_buf0_3[0] = normal_reduce_temp0_1[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int)threadIdx.x) % 32) == 0) {
    red_buf_staging_1[(((int)threadIdx.x) >> 5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    red_buf0_2[0] = red_buf_staging_1[((int)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)15);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int)threadIdx.x) == 0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_softmax_expsum[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
    if (((i2_outer * 16) + (((int)threadIdx.x) >> 3)) < 125) {
      T_softmax_norm[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx.x))] = (__expf((data[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx.x))] - T_softmax_maxelem[0])) / T_softmax_expsum[0]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(128) softmax_shared_2060_copy(float* __restrict__ T_softmax_norm, float* __restrict__ data) {
  float normal_reduce_temp0[1];
  __shared__ float red_result[1+512];//
  __shared__ float T_softmax_maxelem[1+512];//
  float normal_reduce_temp0_1[1];
  __shared__ float red_result_1[1+512];//
  __shared__ float T_softmax_expsum[1+512];//
  normal_reduce_temp0[0] = -3.402823e+38f;
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[((((int)blockIdx.x) * 1000) + ((int)threadIdx.x))]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 128)]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 256)]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 384)]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 512)]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 640)]);
  normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 768)]);
  if (((int)threadIdx.x) < 104) {
    normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 896)]);
  }
  float red_buf0[1];
  uint mask[1];
  float t0[1];
  float red_buf0_1[1];
  uint mask_1[1];
  float t0_1[1];
  __shared__ float red_buf_staging[4];
  red_buf0_1[0] = normal_reduce_temp0[0];
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
  if ((((int)threadIdx.x) % 32) == 0) {
    red_buf_staging[(((int)threadIdx.x) >> 5)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    red_buf0[0] = red_buf_staging[((int)threadIdx.x)];
  }
  mask[0] = (__activemask() & (uint)15);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = max(red_buf0[0], t0[0]);
  if (((int)threadIdx.x) == 0) {
    ((volatile float*)red_result)[0] = red_buf0[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_softmax_maxelem[0] = ((volatile float*)red_result)[0];
  }
  normal_reduce_temp0_1[0] = 0.000000e+00f;
  __syncthreads();
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    if (((k_outer * 16) + (((int)threadIdx.x) >> 3)) < 125) {
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + (k_outer * 128)) + ((int)threadIdx.x))] - T_softmax_maxelem[0])));
    }
  }
  float red_buf0_2[1];
  uint mask_2[1];
  float t0_2[1];
  float red_buf0_3[1];
  uint mask_3[1];
  float t0_3[1];
  __shared__ float red_buf_staging_1[4];
  red_buf0_3[0] = normal_reduce_temp0_1[0];
  mask_3[0] = __activemask();
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
  red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
  if ((((int)threadIdx.x) % 32) == 0) {
    red_buf_staging_1[(((int)threadIdx.x) >> 5)] = red_buf0_3[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 4) {
    red_buf0_2[0] = red_buf_staging_1[((int)threadIdx.x)];
  }
  mask_2[0] = (__activemask() & (uint)15);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
  red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
  if (((int)threadIdx.x) == 0) {
    ((volatile float*)red_result_1)[0] = red_buf0_2[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_softmax_expsum[0] = ((volatile float*)red_result_1)[0];
  }
  __syncthreads();
  for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
    if (((i2_outer * 16) + (((int)threadIdx.x) >> 3)) < 125) {
      T_softmax_norm[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx.x))] = (__expf((data[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx.x))] - T_softmax_maxelem[0])) / T_softmax_expsum[0]);
    }
  }
}