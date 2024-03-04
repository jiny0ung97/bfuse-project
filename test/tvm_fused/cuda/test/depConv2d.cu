
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
extern "C" __global__ void __launch_bounds__(112) depConv2d(float* __restrict__ DepthwiseConv2d, float* __restrict__ data, float* __restrict__ kernel) {
  __shared__ float PaddedInput_shared[3249];
  __shared__ float kernel_shared[9];
  float PaddedInput_shared_local[45];
  float kernel_shared_local[9];
  float DepthwiseConv2d_local[7];
  #pragma unroll
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 30; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
    if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) < 3249) {
      if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 4) + ((int)threadIdx.y)) < 117) {
        PaddedInput_shared[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x))] = (((57 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x))) && (1 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) % 57))) ? data[((((((int)blockIdx.z) * 3136) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) / 57) * 56)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx.y) * 28)) + ((int)threadIdx.x)) % 57)) - 57)] : 0.000000e+00f);
      }
    }
  }
  if (((((int)threadIdx.y) * 28) + ((int)threadIdx.x)) < 9) {
    if (((int)threadIdx.y) < 1) {
      kernel_shared[((((int)threadIdx.y) * 28) + ((int)threadIdx.x))] = kernel[(((((int)threadIdx.y) * 28) + ((((int)blockIdx.z) & 127) * 9)) + ((int)threadIdx.x))];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int ax2 = 0; ax2 < 15; ++ax2) {
    #pragma unroll
    for (int ax3 = 0; ax3 < 3; ++ax3) {
      PaddedInput_shared_local[((ax2 * 3) + ax3)] = PaddedInput_shared[((((((int)threadIdx.y) * 798) + (ax2 * 57)) + (((int)threadIdx.x) * 2)) + ax3)];
    }
  }
  #pragma unroll
  for (int ax2_1 = 0; ax2_1 < 3; ++ax2_1) {
    #pragma unroll
    for (int ax3_1 = 0; ax3_1 < 3; ++ax3_1) {
      kernel_shared_local[((ax2_1 * 3) + ax3_1)] = kernel_shared[((ax2_1 * 3) + ax3_1)];
    }
  }
  #pragma unroll
  for (int i_c = 0; i_c < 7; ++i_c) {
    DepthwiseConv2d_local[i_c] = 0.000000e+00f;
    #pragma unroll
    for (int di = 0; di < 3; ++di) {
      #pragma unroll
      for (int dj = 0; dj < 3; ++dj) {
        DepthwiseConv2d_local[i_c] = (DepthwiseConv2d_local[i_c] + (PaddedInput_shared_local[(((i_c * 6) + (di * 3)) + dj)] * kernel_shared_local[((di * 3) + dj)]));
      }
    }
  }
  #pragma unroll
  for (int i_inner_inner_inner = 0; i_inner_inner_inner < 7; ++i_inner_inner_inner) {
    DepthwiseConv2d[((((((int)blockIdx.z) * 784) + (((int)threadIdx.y) * 196)) + (i_inner_inner_inner * 28)) + ((int)threadIdx.x))] = DepthwiseConv2d_local[i_inner_inner_inner];
  }
}

