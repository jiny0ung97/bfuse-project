
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
extern "C" __global__ void __launch_bounds__(50) default_function_kernel(float* __restrict__ T_softmax_norm, float* __restrict__ data) {
  float normal_reduce_temp0[1];
  __shared__ float red_buf0[50];
  __shared__ float T_softmax_maxelem[1];
  float normal_reduce_temp0_1[1];
  __shared__ float red_buf0_1[50];
  __shared__ float T_softmax_expsum[1];
  normal_reduce_temp0[0] = -3.402823e+38f;
  for (int k_outer = 0; k_outer < 20; ++k_outer) {
    normal_reduce_temp0[0] = max(normal_reduce_temp0[0], data[(((((int)blockIdx.x) * 1000) + (k_outer * 50)) + ((int)threadIdx.x))]);
  }
  __syncthreads();
  ((volatile float*)red_buf0)[((int)threadIdx.x)] = normal_reduce_temp0[0];
  __syncthreads();
  if (((int)threadIdx.x) < 18) {
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = max(((volatile float*)red_buf0)[((int)threadIdx.x)], ((volatile float*)red_buf0)[(((int)threadIdx.x) + 32)]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    float w_16_0 = max(((volatile float*)red_buf0)[((int)threadIdx.x)], ((volatile float*)red_buf0)[(((int)threadIdx.x) + 16)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_16_0;
    float w_8_0 = max(((volatile float*)red_buf0)[((int)threadIdx.x)], ((volatile float*)red_buf0)[(((int)threadIdx.x) + 8)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_8_0;
    float w_4_0 = max(((volatile float*)red_buf0)[((int)threadIdx.x)], ((volatile float*)red_buf0)[(((int)threadIdx.x) + 4)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_4_0;
    float w_2_0 = max(((volatile float*)red_buf0)[((int)threadIdx.x)], ((volatile float*)red_buf0)[(((int)threadIdx.x) + 2)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_2_0;
    float w_1_0 = max(((volatile float*)red_buf0)[((int)threadIdx.x)], ((volatile float*)red_buf0)[(((int)threadIdx.x) + 1)]);
    ((volatile float*)red_buf0)[((int)threadIdx.x)] = w_1_0;
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_softmax_maxelem[0] = ((volatile float*)red_buf0)[0];
  }
  normal_reduce_temp0_1[0] = 0.000000e+00f;
  __syncthreads();
  for (int k_outer_1 = 0; k_outer_1 < 20; ++k_outer_1) {
    normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + (k_outer_1 * 50)) + ((int)threadIdx.x))] - T_softmax_maxelem[0])));
  }
  __syncthreads();
  ((volatile float*)red_buf0_1)[((int)threadIdx.x)] = normal_reduce_temp0_1[0];
  __syncthreads();
  if (((int)threadIdx.x) < 18) {
    ((volatile float*)red_buf0_1)[((int)threadIdx.x)] = (((volatile float*)red_buf0_1)[((int)threadIdx.x)] + ((volatile float*)red_buf0_1)[(((int)threadIdx.x) + 32)]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    float w_16_0_1 = (((volatile float*)red_buf0_1)[((int)threadIdx.x)] + ((volatile float*)red_buf0_1)[(((int)threadIdx.x) + 16)]);
    ((volatile float*)red_buf0_1)[((int)threadIdx.x)] = w_16_0_1;
    float w_8_0_1 = (((volatile float*)red_buf0_1)[((int)threadIdx.x)] + ((volatile float*)red_buf0_1)[(((int)threadIdx.x) + 8)]);
    ((volatile float*)red_buf0_1)[((int)threadIdx.x)] = w_8_0_1;
    float w_4_0_1 = (((volatile float*)red_buf0_1)[((int)threadIdx.x)] + ((volatile float*)red_buf0_1)[(((int)threadIdx.x) + 4)]);
    ((volatile float*)red_buf0_1)[((int)threadIdx.x)] = w_4_0_1;
    float w_2_0_1 = (((volatile float*)red_buf0_1)[((int)threadIdx.x)] + ((volatile float*)red_buf0_1)[(((int)threadIdx.x) + 2)]);
    ((volatile float*)red_buf0_1)[((int)threadIdx.x)] = w_2_0_1;
    float w_1_0_1 = (((volatile float*)red_buf0_1)[((int)threadIdx.x)] + ((volatile float*)red_buf0_1)[(((int)threadIdx.x) + 1)]);
    ((volatile float*)red_buf0_1)[((int)threadIdx.x)] = w_1_0_1;
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    T_softmax_expsum[0] = ((volatile float*)red_buf0_1)[0];
  }
  __syncthreads();
  for (int i2_outer = 0; i2_outer < 20; ++i2_outer) {
    T_softmax_norm[(((((int)blockIdx.x) * 1000) + (i2_outer * 50)) + ((int)threadIdx.x))] = (__expf((data[(((((int)blockIdx.x) * 1000) + (i2_outer * 50)) + ((int)threadIdx.x))] - T_softmax_maxelem[0])) / T_softmax_expsum[0]);
  }
}

