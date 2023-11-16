
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
extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ data, float* __restrict__ pool_max) {
  pool_max[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      pool_max[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = max(pool_max[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))], data[(((((((((int)blockIdx.x) / 729) * 200704) + (((((int)blockIdx.x) % 729) / 27) * 7168)) + (rv0 * 3584)) + ((((int)blockIdx.x) % 27) * 128)) + (rv1 * 64)) + ((int)threadIdx.x))]);
    }
  }
}

