
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
extern "C" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  float pool_max_local[1];
  pool_max_local[0] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      pool_max_local[0] = max(pool_max_local[0], data[((((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 729) * 3136) + (((((((int)blockIdx.x) * 295) + ((int)threadIdx.x)) % 729) / 27) * 112)) + (rv0 * 56)) + ((((((int)blockIdx.x) * 25) + ((int)threadIdx.x)) % 27) * 2)) + rv1)]);
    }
  }
  pool_max[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = pool_max_local[0];
}

