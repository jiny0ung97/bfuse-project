
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
extern "C" __global__ void __launch_bounds__(64) default_function_kernel_1(float* __restrict__ pool_avg, float* __restrict__ pool_sum) {
  pool_avg[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (pool_sum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] * 1.111111e-01f);
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_sum) {
  pool_sum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      pool_sum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (pool_sum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] + (((((1 <= (((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 1225) / 35) + rv0)) && ((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 1225) / 35) + rv0) < 36)) && (1 <= (rv1 + (((((int)blockIdx.x) * 29) + ((int)threadIdx.x)) % 35)))) && ((rv1 + (((((int)blockIdx.x) * 29) + ((int)threadIdx.x)) % 35)) < 36)) ? data[(((((((int)blockIdx.x) * 64) + (rv0 * 35)) + ((int)threadIdx.x)) + rv1) - 36)] : 0.000000e+00f));
    }
  }
}

