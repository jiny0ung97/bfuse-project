
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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[4];
  __shared__ float A_shared[32];
  __shared__ float B_shared[1024];
  T_batch_matmul_NT_local[0] = 0.000000e+00f;
  T_batch_matmul_NT_local[2] = 0.000000e+00f;
  T_batch_matmul_NT_local[1] = 0.000000e+00f;
  T_batch_matmul_NT_local[3] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 16; ++k_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 16) {
      *(float2*)(A_shared + (((int)threadIdx.x) * 2)) = *(float2*)(A + ((((((int)blockIdx.x) >> 5) * 512) + (k_outer_outer * 32)) + (((int)threadIdx.x) * 2)));
    }
    B_shared[((int)threadIdx.x)] = B[((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    B_shared[(((int)threadIdx.x) + 128)] = B[(((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2048)];
    B_shared[(((int)threadIdx.x) + 256)] = B[(((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
    B_shared[(((int)threadIdx.x) + 384)] = B[(((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 6144)];
    B_shared[(((int)threadIdx.x) + 512)] = B[(((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192)];
    B_shared[(((int)threadIdx.x) + 640)] = B[(((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 10240)];
    B_shared[(((int)threadIdx.x) + 768)] = B[(((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288)];
    B_shared[(((int)threadIdx.x) + 896)] = B[(((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.x) >> 5) * 512)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336)];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
      if (((int)threadIdx.x) < 16) {
        T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (A_shared[k_outer_inner] * B_shared[((((int)threadIdx.x) * 64) + k_outer_inner)]));
      }
      if (((int)threadIdx.x) < 16) {
        T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (A_shared[k_outer_inner] * B_shared[(((((int)threadIdx.x) * 64) + k_outer_inner) + 32)]));
      }
    }
  }
  for (int j_inner = 0; j_inner < 2; ++j_inner) {
    if (((int)threadIdx.x) < 16) {
      T_batch_matmul_NT[(((((int)blockIdx.x) * 32) + (((int)threadIdx.x) * 2)) + j_inner)] = T_batch_matmul_NT_local[j_inner];
    }
  }
}

