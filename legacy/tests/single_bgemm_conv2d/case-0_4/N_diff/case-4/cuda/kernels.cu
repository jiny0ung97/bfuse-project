extern "C" __global__ void __launch_bounds__(64) bgemm_0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[64];
  __shared__ float A_shared[512];
  __shared__ float B_shared[512];
  float A_shared_local[8];
  float B_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      T_batch_matmul_NT_local[((i_c_init * 8) + j_c_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    __syncthreads();
    A_shared[((((int)threadIdx.y) * 64) + ((int)threadIdx.x))] = A[(((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x))];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 8)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 64)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 16)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 128)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 24)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 192)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 32)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 256)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 40)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 320)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 48)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 384)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 56)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 448)];
    B_shared[((((int)threadIdx.y) * 64) + ((int)threadIdx.x))] = B[(((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x))];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 8)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 64)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 16)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 128)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 24)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 192)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 32)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 256)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 40)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 320)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 48)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 384)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 56)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 448)];
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      A_shared_local[0] = A_shared[((((int)threadIdx.y) * 64) + k_inner)];
      A_shared_local[1] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 8)];
      A_shared_local[2] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 16)];
      A_shared_local[3] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 24)];
      A_shared_local[4] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 32)];
      A_shared_local[5] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 40)];
      A_shared_local[6] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 48)];
      A_shared_local[7] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 56)];
      B_shared_local[0] = B_shared[((((int)threadIdx.x) * 64) + k_inner)];
      B_shared_local[1] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 8)];
      B_shared_local[2] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 16)];
      B_shared_local[3] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 24)];
      B_shared_local[4] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 32)];
      B_shared_local[5] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 40)];
      B_shared_local[6] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 48)];
      B_shared_local[7] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 56)];
      for (int i_c = 0; i_c < 8; ++i_c) {
        T_batch_matmul_NT_local[(i_c * 8)] = (T_batch_matmul_NT_local[(i_c * 8)] + (A_shared_local[i_c] * B_shared_local[0]));
        T_batch_matmul_NT_local[((i_c * 8) + 1)] = (T_batch_matmul_NT_local[((i_c * 8) + 1)] + (A_shared_local[i_c] * B_shared_local[1]));
        T_batch_matmul_NT_local[((i_c * 8) + 2)] = (T_batch_matmul_NT_local[((i_c * 8) + 2)] + (A_shared_local[i_c] * B_shared_local[2]));
        T_batch_matmul_NT_local[((i_c * 8) + 3)] = (T_batch_matmul_NT_local[((i_c * 8) + 3)] + (A_shared_local[i_c] * B_shared_local[3]));
        T_batch_matmul_NT_local[((i_c * 8) + 4)] = (T_batch_matmul_NT_local[((i_c * 8) + 4)] + (A_shared_local[i_c] * B_shared_local[4]));
        T_batch_matmul_NT_local[((i_c * 8) + 5)] = (T_batch_matmul_NT_local[((i_c * 8) + 5)] + (A_shared_local[i_c] * B_shared_local[5]));
        T_batch_matmul_NT_local[((i_c * 8) + 6)] = (T_batch_matmul_NT_local[((i_c * 8) + 6)] + (A_shared_local[i_c] * B_shared_local[6]));
        T_batch_matmul_NT_local[((i_c * 8) + 7)] = (T_batch_matmul_NT_local[((i_c * 8) + 7)] + (A_shared_local[i_c] * B_shared_local[7]));
      }
    }
  }
  for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
    T_batch_matmul_NT[((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8))] = T_batch_matmul_NT_local[(i_inner_inner * 8)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 1)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 1)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 2)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 2)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 3)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 3)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 4)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 4)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 5)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 5)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 6)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 6)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 7)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 7)];
  }
}

extern "C" __global__ void __launch_bounds__(152) conv2d_4(float* __restrict__ conv2d_nchw, float* __restrict__ data, float* __restrict__ kernel) {
  float conv2d_nchw_local[4];
  __shared__ float pad_temp_shared[450];
  __shared__ float kernel_shared[576];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if (((((int)threadIdx.z) * 19) + ((int)threadIdx.x)) < 150) {
        pad_temp_shared[(((((int)threadIdx.z) * 57) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = data[((((((((((int)blockIdx.z) >> 1) * 824464) + (rc_outer * 103058)) + ((((((int)threadIdx.z) * 19) + ((int)threadIdx.x)) / 75) * 51529)) + (((int)blockIdx.y) * 908)) + (((((((int)threadIdx.z) * 19) + ((int)threadIdx.x)) % 75) / 25) * 227)) + (((int)blockIdx.x) * 76)) + ((((((int)threadIdx.z) * 57) + (((int)threadIdx.x) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 75))];
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) {
      if (((((int)threadIdx.x) / 18) + ((int)threadIdx.z)) < 8) {
        if (((int)threadIdx.x) < 18) {
          kernel_shared[(((((int)threadIdx.z) * 72) + (((int)threadIdx.x) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1)] = kernel[((((((((int)blockIdx.z) & 1) * 4608) + (((int)threadIdx.z) * 576)) + ((((((int)threadIdx.x) * 2) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 >> 1)) / 9) * 144)) + (rc_outer * 18)) + (((((int)threadIdx.x) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) % 18))];
        }
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
      for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
        for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_inner * 225) + (ry_inner * 75)) + (((int)threadIdx.x) * 4)) + rx_inner)] * kernel_shared[((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((rc_inner * 225) + (ry_inner * 75)) + (((int)threadIdx.x) * 4)) + rx_inner)] * kernel_shared[(((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 144)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((rc_inner * 225) + (ry_inner * 75)) + (((int)threadIdx.x) * 4)) + rx_inner)] * kernel_shared[(((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 288)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((rc_inner * 225) + (ry_inner * 75)) + (((int)threadIdx.x) * 4)) + rx_inner)] * kernel_shared[(((((((int)threadIdx.z) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 432)]));
        }
      }
    }
  }
  conv2d_nchw[(((((((int)blockIdx.z) * 103968) + (((int)threadIdx.z) * 3249)) + (((int)blockIdx.y) * 57)) + (((int)blockIdx.x) * 19)) + ((int)threadIdx.x))] = conv2d_nchw_local[0];
  conv2d_nchw[((((((((int)blockIdx.z) * 103968) + (((int)threadIdx.z) * 3249)) + (((int)blockIdx.y) * 57)) + (((int)blockIdx.x) * 19)) + ((int)threadIdx.x)) + 25992)] = conv2d_nchw_local[1];
  conv2d_nchw[((((((((int)blockIdx.z) * 103968) + (((int)threadIdx.z) * 3249)) + (((int)blockIdx.y) * 57)) + (((int)blockIdx.x) * 19)) + ((int)threadIdx.x)) + 51984)] = conv2d_nchw_local[2];
  conv2d_nchw[((((((((int)blockIdx.z) * 103968) + (((int)threadIdx.z) * 3249)) + (((int)blockIdx.y) * 57)) + (((int)blockIdx.x) * 19)) + ((int)threadIdx.x)) + 77976)] = conv2d_nchw_local[3];
}

