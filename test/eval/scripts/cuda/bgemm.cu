
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
extern "C" __global__ void __launch_bounds__(8) default_function_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[1];
  __shared__ float A_shared[8];
  __shared__ float B_shared[64];
  float A_shared_local[1];
  float B_shared_local[1];
  T_batch_matmul_NT_local[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    __syncthreads();
    A_shared[((int)threadIdx.x)] = A[(((((int)blockIdx.z) * 512) + (k_outer * 8)) + ((int)threadIdx.x))];
    #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      B_shared[((ax1_inner * 8) + ((int)threadIdx.x))] = B[(((((((int)blockIdx.z) * 512000) + (((int)blockIdx.x) * 4096)) + (ax1_inner * 512)) + (k_outer * 8)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      A_shared_local[0] = A_shared[k_inner];
      B_shared_local[0] = B_shared[((((int)threadIdx.x) * 8) + k_inner)];
      T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (A_shared_local[0] * B_shared_local[0]));
    }
  }
  T_batch_matmul_NT[(((((int)blockIdx.z) * 1000) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))] = T_batch_matmul_NT_local[0];
}

