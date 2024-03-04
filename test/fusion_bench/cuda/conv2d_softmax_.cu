
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
extern "C" __global__ void __launch_bounds__(232) conv2d_test_2(float* __restrict__ conv2d_nchw, float* __restrict__ data, float* __restrict__ kernel) {
  float conv2d_nchw_local[16];
  __shared__ float pad_temp_shared[7424];
  __shared__ float kernel_shared[2048];
  for (int ff_c_inner_init = 0; ff_c_inner_init < 2; ++ff_c_inner_init) {
    conv2d_nchw_local[ff_c_inner_init] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_inner_init + 2)] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_inner_init + 4)] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_inner_init + 6)] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_inner_init + 8)] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_inner_init + 10)] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_inner_init + 12)] = 0.000000e+00f;
    conv2d_nchw_local[(ff_c_inner_init + 14)] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 32; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
    pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 232) + ((int)threadIdx.x))] = (((((1 <= (((((int)blockIdx.x) % 29) * 2) + ((((int)threadIdx.x) % 116) / 58))) && ((((((int)blockIdx.x) % 29) * 2) + ((((int)threadIdx.x) % 116) / 58)) < 57)) && (1 <= (((int)threadIdx.x) % 58))) && ((((int)threadIdx.x) % 58) < 57)) ? data[((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 6272) + ((((int)threadIdx.x) / 116) * 3136)) + ((((int)blockIdx.x) % 29) * 112)) + (((((int)threadIdx.x) % 116) / 58) * 56)) + (((int)threadIdx.x) % 58)) - 57)] : 0.000000e+00f);
  }
  for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1) {
    if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 29) + (((int)threadIdx.x) >> 3)) < 128) {
      *(float2*)(kernel_shared + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 464) + (((int)threadIdx.x) * 2))) = *(float2*)(kernel + ((((((int)blockIdx.x) / 29) * 2048) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 464)) + (((int)threadIdx.x) * 2)));
    }
  }
  __syncthreads();
  for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      for (int ff_c_inner = 0; ff_c_inner < 2; ++ff_c_inner) {
        conv2d_nchw_local[ff_c_inner] = (conv2d_nchw_local[ff_c_inner] + (pad_temp_shared[(((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx.x) % 58))] * kernel_shared[(((((((int)threadIdx.x) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw_local[(ff_c_inner + 2)] = (conv2d_nchw_local[(ff_c_inner + 2)] + (pad_temp_shared[((((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx.x) % 58)) + 58)] * kernel_shared[(((((((int)threadIdx.x) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner)]));
        conv2d_nchw_local[(ff_c_inner + 4)] = (conv2d_nchw_local[(ff_c_inner + 4)] + (pad_temp_shared[(((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx.x) % 58))] * kernel_shared[((((((((int)threadIdx.x) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 512)]));
        conv2d_nchw_local[(ff_c_inner + 6)] = (conv2d_nchw_local[(ff_c_inner + 6)] + (pad_temp_shared[((((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx.x) % 58)) + 58)] * kernel_shared[((((((((int)threadIdx.x) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 512)]));
        conv2d_nchw_local[(ff_c_inner + 8)] = (conv2d_nchw_local[(ff_c_inner + 8)] + (pad_temp_shared[(((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx.x) % 58))] * kernel_shared[((((((((int)threadIdx.x) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 1024)]));
        conv2d_nchw_local[(ff_c_inner + 10)] = (conv2d_nchw_local[(ff_c_inner + 10)] + (pad_temp_shared[((((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx.x) % 58)) + 58)] * kernel_shared[((((((((int)threadIdx.x) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 1024)]));
        conv2d_nchw_local[(ff_c_inner + 12)] = (conv2d_nchw_local[(ff_c_inner + 12)] + (pad_temp_shared[(((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx.x) % 58))] * kernel_shared[((((((((int)threadIdx.x) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 1536)]));
        conv2d_nchw_local[(ff_c_inner + 14)] = (conv2d_nchw_local[(ff_c_inner + 14)] + (pad_temp_shared[((((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx.x) % 58)) + 58)] * kernel_shared[((((((((int)threadIdx.x) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 1536)]));
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    conv2d_nchw[((((((((int)blockIdx.x) / 29) * 107648) + ((((int)threadIdx.x) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx.x) % 29) * 116)) + (((int)threadIdx.x) % 58))] = conv2d_nchw_local[ff_inner];
    conv2d_nchw[(((((((((int)blockIdx.x) / 29) * 107648) + ((((int)threadIdx.x) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx.x) % 29) * 116)) + (((int)threadIdx.x) % 58)) + 58)] = conv2d_nchw_local[(ff_inner + 2)];
    conv2d_nchw[(((((((((int)blockIdx.x) / 29) * 107648) + ((((int)threadIdx.x) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx.x) % 29) * 116)) + (((int)threadIdx.x) % 58)) + 26912)] = conv2d_nchw_local[(ff_inner + 4)];
    conv2d_nchw[(((((((((int)blockIdx.x) / 29) * 107648) + ((((int)threadIdx.x) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx.x) % 29) * 116)) + (((int)threadIdx.x) % 58)) + 26970)] = conv2d_nchw_local[(ff_inner + 6)];
    conv2d_nchw[(((((((((int)blockIdx.x) / 29) * 107648) + ((((int)threadIdx.x) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx.x) % 29) * 116)) + (((int)threadIdx.x) % 58)) + 53824)] = conv2d_nchw_local[(ff_inner + 8)];
    conv2d_nchw[(((((((((int)blockIdx.x) / 29) * 107648) + ((((int)threadIdx.x) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx.x) % 29) * 116)) + (((int)threadIdx.x) % 58)) + 53882)] = conv2d_nchw_local[(ff_inner + 10)];
    conv2d_nchw[(((((((((int)blockIdx.x) / 29) * 107648) + ((((int)threadIdx.x) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx.x) % 29) * 116)) + (((int)threadIdx.x) % 58)) + 80736)] = conv2d_nchw_local[(ff_inner + 12)];
    conv2d_nchw[(((((((((int)blockIdx.x) / 29) * 107648) + ((((int)threadIdx.x) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx.x) % 29) * 116)) + (((int)threadIdx.x) % 58)) + 80794)] = conv2d_nchw_local[(ff_inner + 14)];
  }
}

extern "C" __global__ void __launch_bounds__(50) softmax_test(float* __restrict__ T_softmax_norm, float* __restrict__ data) {
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
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[((((int)blockIdx.x) * 1000) + ((int)threadIdx.x))] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 50)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 100)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 150)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 200)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 250)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 300)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 350)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 400)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 450)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 500)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 550)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 600)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 650)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 700)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 750)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 800)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 850)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 900)] - T_softmax_maxelem[0])));
  normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((data[(((((int)blockIdx.x) * 1000) + ((int)threadIdx.x)) + 950)] - T_softmax_maxelem[0])));
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
  for (int i1_outer = 0; i1_outer < 20; ++i1_outer) {
    T_softmax_norm[(((((int)blockIdx.x) * 1000) + (i1_outer * 50)) + ((int)threadIdx.x))] = (__expf((data[(((((int)blockIdx.x) * 1000) + (i1_outer * 50)) + ((int)threadIdx.x))] - T_softmax_maxelem[0])) / T_softmax_expsum[0]);
  }
}

