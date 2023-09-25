
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
extern "C" __global__ void __launch_bounds__(128) conv2d_B8(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nhwc) {
  float conv2d_nhwc_local[16];
  __shared__ float pad_temp_shared[480];
  __shared__ float kernel_shared[3072];
  for (int nn_c_outer_inner_init = 0; nn_c_outer_inner_init < 2; ++nn_c_outer_inner_init) {
    for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 2; ++ff_c_outer_inner_init) {
      for (int nn_c_inner_init = 0; nn_c_inner_init < 2; ++nn_c_inner_init) {
        conv2d_nhwc_local[(((nn_c_outer_inner_init * 4) + (nn_c_inner_init * 2)) + ff_c_outer_inner_init)] = 0.000000e+00f;
        conv2d_nhwc_local[((((nn_c_outer_inner_init * 4) + (nn_c_inner_init * 2)) + ff_c_outer_inner_init) + 8)] = 0.000000e+00f;
      }
    }
  }
  for (int ry_outer_outer = 0; ry_outer_outer < 3; ++ry_outer_outer) {
    for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
      __syncthreads();
      if (((int)threadIdx.x) < 120) {
        *(float4*)(pad_temp_shared + (((int)threadIdx.x) * 4)) = (((1 <= (((((((int)blockIdx.x) % 196) / 14) * 4) + ((((int)threadIdx.x) % 30) / 10)) + ry_outer_outer)) && (1 <= (((((int)blockIdx.x) % 14) * 4) + ((((int)threadIdx.x) % 10) >> 1)))) ? *(float4*)(data + (((((((((((((int)blockIdx.x) / 196) * 802816) + ((((int)threadIdx.x) / 30) * 200704)) + (((((int)blockIdx.x) % 196) / 14) * 14336)) + (((((int)threadIdx.x) % 30) / 10) * 3584)) + (ry_outer_outer * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + (((((int)threadIdx.x) % 10) >> 1) * 64)) + (rc_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) - 3648)) : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
      }
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
        *(float4*)(kernel_shared + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(kernel + (((((ry_outer_outer * 24576) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer >> 1) * 8192)) + (rc_outer_outer * 1024)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 1) * 512)) + (((int)threadIdx.x) * 4)));
      }
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
        for (int nn_c_outer_inner = 0; nn_c_outer_inner < 2; ++nn_c_outer_inner) {
          for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
            for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
              for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
                for (int nn_c_inner = 0; nn_c_inner < 2; ++nn_c_inner) {
                  conv2d_nhwc_local[(((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner)] = (conv2d_nhwc_local[(((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner)] + (pad_temp_shared[((((((nn_c_outer_inner * 240) + (nn_c_inner * 120)) + ((((int)threadIdx.x) >> 6) * 80)) + (rx_inner * 8)) + (rc_outer_inner * 2)) + rc_inner)] * kernel_shared[(((((rx_inner * 1024) + (rc_outer_inner * 256)) + (rc_inner * 128)) + ((((int)threadIdx.x) & 63) * 2)) + ff_c_outer_inner)]));
                  conv2d_nhwc_local[((((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner) + 8)] = (conv2d_nhwc_local[((((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner) + 8)] + (pad_temp_shared[(((((((nn_c_outer_inner * 240) + (nn_c_inner * 120)) + ((((int)threadIdx.x) >> 6) * 80)) + (rx_inner * 8)) + (rc_outer_inner * 2)) + rc_inner) + 16)] * kernel_shared[(((((rx_inner * 1024) + (rc_outer_inner * 256)) + (rc_inner * 128)) + ((((int)threadIdx.x) & 63) * 2)) + ff_c_outer_inner)]));
                }
              }
            }
          }
        }
      }
    }
  }
  for (int nn_inner = 0; nn_inner < 4; ++nn_inner) {
    for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
      conv2d_nhwc[((((((((((int)blockIdx.x) / 196) * 401408) + (nn_inner * 100352)) + (((((int)blockIdx.x) % 196) / 14) * 7168)) + ((((int)threadIdx.x) >> 6) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + ff_inner)] = conv2d_nhwc_local[((nn_inner * 2) + ff_inner)];
      conv2d_nhwc[(((((((((((int)blockIdx.x) / 196) * 401408) + (nn_inner * 100352)) + (((((int)blockIdx.x) % 196) / 14) * 7168)) + ((((int)threadIdx.x) >> 6) * 3584)) + ((((int)blockIdx.x) % 14) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + ff_inner) + 128)] = conv2d_nhwc_local[(((nn_inner * 2) + ff_inner) + 8)];
    }
  }
}

