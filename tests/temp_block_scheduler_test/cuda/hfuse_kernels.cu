#include <stdio.h>

extern "C" __global__ __launch_bounds__(96) void bgemm_0_conv2d_0_fused_hfuse(float *__restrict bgemm_0_A_, float *__restrict bgemm_0_B_, float *__restrict bgemm_0_T_batch_matmul_NT_, float *__restrict conv2d_0_conv2d_nchw_, float *__restrict conv2d_0_data_, float *__restrict conv2d_0_kernel_)
{
  uint streamingMultiprocessorId;
  asm("mov.u32 %0, %smid;" : "=r"(streamingMultiprocessorId));
  printf("Block: %d | SM: %d - Here!\n", blockIdx.x, streamingMultiprocessorId);

  // bgemm_0
  if (((int)threadIdx.x >= 0 && (int)threadIdx.x < 64) && ((int)blockIdx.x >= 0 && (int)blockIdx.x < 1024))
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
  // conv2d_0
  if (((int)threadIdx.x >= 64 && (int)threadIdx.x < 80) && ((int)blockIdx.x >= 0 && (int)blockIdx.x < 1815))
  {
      int blockIdx_x_ = (int)blockIdx.x % 55;
      int blockIdx_y_ = (int)blockIdx.x / 55 % 11;
      int blockIdx_z_ = (int)blockIdx.x / 605;
      int threadIdx_x_ = ((int)threadIdx.x - 64) % 1;
      int threadIdx_y_ = ((int)threadIdx.x - 64) / 1 % 1;
      int threadIdx_z_ = ((int)threadIdx.x - 64) / 1;

      float conv2d_nchw_local[20];
      static float pad_temp_shared[18] __attribute__((shared));
      static float kernel_shared[128] __attribute__((shared));
      conv2d_nchw_local[0] = 0.F;
      conv2d_nchw_local[10] = 0.F;
      conv2d_nchw_local[2] = 0.F;
      conv2d_nchw_local[12] = 0.F;
      conv2d_nchw_local[4] = 0.F;
      conv2d_nchw_local[14] = 0.F;
      conv2d_nchw_local[6] = 0.F;
      conv2d_nchw_local[16] = 0.F;
      conv2d_nchw_local[8] = 0.F;
      conv2d_nchw_local[18] = 0.F;
      conv2d_nchw_local[1] = 0.F;
      conv2d_nchw_local[11] = 0.F;
      conv2d_nchw_local[3] = 0.F;
      conv2d_nchw_local[13] = 0.F;
      conv2d_nchw_local[5] = 0.F;
      conv2d_nchw_local[15] = 0.F;
      conv2d_nchw_local[7] = 0.F;
      conv2d_nchw_local[17] = 0.F;
      conv2d_nchw_local[9] = 0.F;
      conv2d_nchw_local[19] = 0.F;
      for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
          asm ("bar.sync 2, 32;");
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[(((((rc_outer * 25088) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2))];
          }
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[(((((rc_outer * 25088) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2))];
          }
          kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[(((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18))];
          kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 9)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 576)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 585)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1152)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1161)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1728)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1737)];
          asm ("bar.sync 2, 32;");
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          asm ("bar.sync 2, 32;");
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((rc_outer * 25088) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 1)];
          }
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((rc_outer * 25088) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 1)];
          }
          kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 10)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 577)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 586)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1153)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1162)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1729)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1738)];
          asm ("bar.sync 2, 32;");
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          asm ("bar.sync 2, 32;");
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((rc_outer * 25088) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 2)];
          }
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((rc_outer * 25088) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 2)];
          }
          kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 2)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 11)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 578)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 587)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1154)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1163)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1730)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1739)];
          asm ("bar.sync 2, 32;");
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          asm ("bar.sync 2, 32;");
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((rc_outer * 25088) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 112)];
          }
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((rc_outer * 25088) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 112)];
          }
          kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 3)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 12)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 579)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 588)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1155)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1164)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1731)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1740)];
          asm ("bar.sync 2, 32;");
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          asm ("bar.sync 2, 32;");
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((rc_outer * 25088) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 113)];
          }
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((rc_outer * 25088) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 113)];
          }
          kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 4)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 13)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 580)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 589)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1156)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1165)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1732)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1741)];
          asm ("bar.sync 2, 32;");
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          asm ("bar.sync 2, 32;");
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((rc_outer * 25088) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 114)];
          }
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((rc_outer * 25088) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 114)];
          }
          kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 5)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 14)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 581)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 590)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1157)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1166)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1733)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1742)];
          asm ("bar.sync 2, 32;");
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          asm ("bar.sync 2, 32;");
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((rc_outer * 25088) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 224)];
          }
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((rc_outer * 25088) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 224)];
          }
          kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 6)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 15)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 582)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 591)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1158)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1167)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1734)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1743)];
          asm ("bar.sync 2, 32;");
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          asm ("bar.sync 2, 32;");
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((rc_outer * 25088) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 225)];
          }
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((rc_outer * 25088) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 225)];
          }
          kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 7)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 16)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 583)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 592)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1159)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1168)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1735)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1744)];
          asm ("bar.sync 2, 32;");
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          asm ("bar.sync 2, 32;");
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((rc_outer * 25088) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 226)];
          }
          if (((int)threadIdx_z_) < 9) {
              pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((rc_outer * 25088) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 226)];
          }
          kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 8)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 17)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 584)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 593)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1160)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1169)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1736)];
          kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[((((((int)blockIdx_z_) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1745)];
          asm ("bar.sync 2, 32;");
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
      }
      conv2d_0_conv2d_nchw_[((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_))] = conv2d_nchw_local[0];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 96800)] = conv2d_nchw_local[10];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 55)] = conv2d_nchw_local[2];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 96855)] = conv2d_nchw_local[12];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 110)] = conv2d_nchw_local[4];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 96910)] = conv2d_nchw_local[14];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 165)] = conv2d_nchw_local[6];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 96965)] = conv2d_nchw_local[16];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 220)] = conv2d_nchw_local[8];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 97020)] = conv2d_nchw_local[18];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 3025)] = conv2d_nchw_local[1];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 99825)] = conv2d_nchw_local[11];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 3080)] = conv2d_nchw_local[3];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 99880)] = conv2d_nchw_local[13];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 3135)] = conv2d_nchw_local[5];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 99935)] = conv2d_nchw_local[15];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 3190)] = conv2d_nchw_local[7];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 99990)] = conv2d_nchw_local[17];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 3245)] = conv2d_nchw_local[9];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 100045)] = conv2d_nchw_local[19];
  }
}
