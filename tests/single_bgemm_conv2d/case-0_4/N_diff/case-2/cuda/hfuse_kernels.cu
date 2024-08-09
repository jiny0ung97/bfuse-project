extern "C" __global__ __launch_bounds__(224) void bgemm_0_conv2d_4_fused_hfuse(float *__restrict bgemm_0_A_, float *__restrict bgemm_0_B_, float *__restrict bgemm_0_T_batch_matmul_NT_, float *__restrict conv2d_4_conv2d_nchw_, float *__restrict conv2d_4_data_, float *__restrict conv2d_4_kernel_)
{
  // bgemm_0
  if (((int)threadIdx.x >= 0 && (int)threadIdx.x < 64) && ((int)blockIdx.x >= 0 && (int)blockIdx.x < 65536))
  {
      int blockIdx_x_ = (int)blockIdx.x % 8;
      int blockIdx_y_ = (int)blockIdx.x / 8 % 8;
      int blockIdx_z_ = (int)blockIdx.x / 64;
      int threadIdx_x_ = ((int)threadIdx.x - 0) % 8;
      int threadIdx_y_ = ((int)threadIdx.x - 0) / 8 % 8;
      int threadIdx_z_ = ((int)threadIdx.x - 0) / 64;

      float T_batch_matmul_NT_local[64];
      static float A_shared[512] __attribute__((shared));
      static float B_shared[512] __attribute__((shared));
      float A_shared_local[8];
      float B_shared_local[8];
      for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
          for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
              T_batch_matmul_NT_local[((i_c_init * 8) + j_c_init)] = 0.F;
          }
      }
      for (int k_outer = 0; k_outer < 8; ++k_outer) {
          asm ("bar.sync 1, 64;");
          A_shared[((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_))] = bgemm_0_A_[(((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_))];
          A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 8)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 64)];
          A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 16)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 128)];
          A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 24)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 192)];
          A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 32)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 256)];
          A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 40)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 320)];
          A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 48)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 384)];
          A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 56)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 448)];
          B_shared[((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_))] = bgemm_0_B_[(((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_))];
          B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 8)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 64)];
          B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 16)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 128)];
          B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 24)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 192)];
          B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 32)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 256)];
          B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 40)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 320)];
          B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 48)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 384)];
          B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 56)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 448)];
          asm ("bar.sync 1, 64;");
          for (int k_inner = 0; k_inner < 8; ++k_inner) {
              A_shared_local[0] = A_shared[((((int)threadIdx_y_) * 64) + k_inner)];
              A_shared_local[1] = A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 8)];
              A_shared_local[2] = A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 16)];
              A_shared_local[3] = A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 24)];
              A_shared_local[4] = A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 32)];
              A_shared_local[5] = A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 40)];
              A_shared_local[6] = A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 48)];
              A_shared_local[7] = A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 56)];
              B_shared_local[0] = B_shared[((((int)threadIdx_x_) * 64) + k_inner)];
              B_shared_local[1] = B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 8)];
              B_shared_local[2] = B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 16)];
              B_shared_local[3] = B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 24)];
              B_shared_local[4] = B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 32)];
              B_shared_local[5] = B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 40)];
              B_shared_local[6] = B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 48)];
              B_shared_local[7] = B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 56)];
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
          bgemm_0_T_batch_matmul_NT_[((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8))] = T_batch_matmul_NT_local[(i_inner_inner * 8)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 1)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 1)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 2)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 2)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 3)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 3)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 4)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 4)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 5)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 5)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 6)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 6)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 7)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 7)];
      }
  }
  // conv2d_4
  if (((int)threadIdx.x >= 64 && (int)threadIdx.x < 216) && ((int)blockIdx.x >= 0 && (int)blockIdx.x < 350208))
  {
      int blockIdx_x_ = (int)blockIdx.x % 3;
      int blockIdx_y_ = (int)blockIdx.x / 3 % 57;
      int blockIdx_z_ = (int)blockIdx.x / 171;
      int threadIdx_x_ = ((int)threadIdx.x - 64) % 19;
      int threadIdx_y_ = ((int)threadIdx.x - 64) / 19 % 1;
      int threadIdx_z_ = ((int)threadIdx.x - 64) / 19;

      float conv2d_nchw_local[4];
      static float pad_temp_shared[450] __attribute__((shared));
      static float kernel_shared[576] __attribute__((shared));
      conv2d_nchw_local[0] = 0.F;
      conv2d_nchw_local[1] = 0.F;
      conv2d_nchw_local[2] = 0.F;
      conv2d_nchw_local[3] = 0.F;
      for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
          asm ("bar.sync 2, 160;");
          for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 3; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
              if (((((int)threadIdx_z_) * 19) + ((int)threadIdx_x_)) < 150) {
                  pad_temp_shared[(((((int)threadIdx_z_) * 57) + (((int)threadIdx_x_) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner)] = conv2d_4_data_[((((((((((int)blockIdx_z_) >> 1) * 824464) + (rc_outer * 103058)) + ((((((int)threadIdx_z_) * 19) + ((int)threadIdx_x_)) / 75) * 51529)) + (((int)blockIdx_y_) * 908)) + (((((((int)threadIdx_z_) * 19) + ((int)threadIdx_x_)) % 75) / 25) * 227)) + (((int)blockIdx_x_) * 76)) + ((((((int)threadIdx_z_) * 57) + (((int)threadIdx_x_) * 3)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 75))];
              }
          }
          for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 < 4; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) {
              if (((((int)threadIdx_x_) / 18) + ((int)threadIdx_z_)) < 8) {
                  if (((int)threadIdx_x_) < 18) {
                      kernel_shared[(((((int)threadIdx_z_) * 72) + (((int)threadIdx_x_) * 4)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1)] = conv2d_4_kernel_[((((((((int)blockIdx_z_) & 1) * 4608) + (((int)threadIdx_z_) * 576)) + ((((((int)threadIdx_x_) * 2) + (ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1 >> 1)) / 9) * 144)) + (rc_outer * 18)) + (((((int)threadIdx_x_) * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner_1) % 18))];
                  }
              }
          }
          asm ("bar.sync 2, 160;");
          for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
              for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
                  for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
                      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_inner * 225) + (ry_inner * 75)) + (((int)threadIdx_x_) * 4)) + rx_inner)] * kernel_shared[((((((int)threadIdx_z_) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner)]));
                      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((rc_inner * 225) + (ry_inner * 75)) + (((int)threadIdx_x_) * 4)) + rx_inner)] * kernel_shared[(((((((int)threadIdx_z_) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 144)]));
                      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((rc_inner * 225) + (ry_inner * 75)) + (((int)threadIdx_x_) * 4)) + rx_inner)] * kernel_shared[(((((((int)threadIdx_z_) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 288)]));
                      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((rc_inner * 225) + (ry_inner * 75)) + (((int)threadIdx_x_) * 4)) + rx_inner)] * kernel_shared[(((((((int)threadIdx_z_) * 18) + (rc_inner * 9)) + (ry_inner * 3)) + rx_inner) + 432)]));
                  }
              }
          }
      }
      conv2d_4_conv2d_nchw_[(((((((int)blockIdx_z_) * 103968) + (((int)threadIdx_z_) * 3249)) + (((int)blockIdx_y_) * 57)) + (((int)blockIdx_x_) * 19)) + ((int)threadIdx_x_))] = conv2d_nchw_local[0];
      conv2d_4_conv2d_nchw_[((((((((int)blockIdx_z_) * 103968) + (((int)threadIdx_z_) * 3249)) + (((int)blockIdx_y_) * 57)) + (((int)blockIdx_x_) * 19)) + ((int)threadIdx_x_)) + 25992)] = conv2d_nchw_local[1];
      conv2d_4_conv2d_nchw_[((((((((int)blockIdx_z_) * 103968) + (((int)threadIdx_z_) * 3249)) + (((int)blockIdx_y_) * 57)) + (((int)blockIdx_x_) * 19)) + ((int)threadIdx_x_)) + 51984)] = conv2d_nchw_local[2];
      conv2d_4_conv2d_nchw_[((((((((int)blockIdx_z_) * 103968) + (((int)threadIdx_z_) * 3249)) + (((int)blockIdx_y_) * 57)) + (((int)blockIdx_x_) * 19)) + ((int)threadIdx_x_)) + 77976)] = conv2d_nchw_local[3];
  }
}
