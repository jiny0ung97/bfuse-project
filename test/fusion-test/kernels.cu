
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
extern "C" __global__ void __launch_bounds__(112) conv2d_B1(float* __restrict__ conv2d_B1_data_, float* __restrict__ conv2d_B1_kernel_, float* __restrict__ conv2d_B1_conv2d_nhwc_) {
  float conv2d_nhwc_local[4];
  __shared__ float pad_temp_shared[2320];
  __shared__ float kernel_shared[2304];

  conv2d_nhwc_local[0] = conv2d_B1_data_[0];

  // int a = blockIdx.x;
  // int b = gridDim.y;
  __syncthreads();
}

extern "C" __global__ void __launch_bounds__(50) matmul_B16(float* __restrict__ matmul_B16_data_, float* __restrict__ matmul_B16_weight_, float* __restrict__ matmul_B16_T_matmul_NT_) {
  float T_matmul_NT_local[4];
  __shared__ float data_shared[256];
  __shared__ float weight_shared[3200];
}