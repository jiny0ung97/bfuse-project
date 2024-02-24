

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

extern "C" __global__ __launch_bounds__(58) void bgemm_conv2d_fused_kernel_bfuse_idx_0(float *__restrict bgemm_A_, float *__restrict bgemm_B_, float *__restrict bgemm_T_batch_matmul_NT_, float *__restrict conv2d_conv2d_nchw_, float *__restrict conv2d_data_, float *__restrict conv2d_kernel_)
{
  /*
   * KernelID_ means...
   * 0: bgemm
   * 1: conv2d
   */
  int gridDim_x_, gridDim_y_, gridDim_z_;
  int blockIdx_x_, blockIdx_y_, blockIdx_z_;
  int blockDim_x_, blockDim_y_, blockDim_z_;
  int threadIdx_x_, threadIdx_y_, threadIdx_z_;
  int NewBlockIdx_;
  int KernelID_;
  
  if (((int)blockIdx.x >= 0 && (int)blockIdx.x < 3696) && ((((int)blockIdx.x - 0) / 84) % 2 == 0))
  {
    NewBlockIdx_ = 0 + (((int)blockIdx.x - 0) / 168) * 84 + (int)blockIdx.x % 84;
    KernelID_  = 1;
    gridDim_x_ = 2;
    gridDim_y_ = 29;
    gridDim_z_ = 32;
    blockDim_x_ = 29;
    blockDim_y_ = 1;
    blockDim_z_ = 2;
  }
  else if (((int)blockIdx.x >= 0 && (int)blockIdx.x < 3696) && ((((int)blockIdx.x - 0) / 84) % 2 == 1))
  {
    NewBlockIdx_ = 0 + (((int)blockIdx.x - 0) / 168) * 84 + (int)blockIdx.x % 84;
    KernelID_  = 0;
    gridDim_x_ = 125;
    gridDim_y_ = 1;
    gridDim_z_ = 32;
    blockDim_x_ = 8;
    blockDim_y_ = 1;
    blockDim_z_ = 1;
  }
  else if (((int)blockIdx.x >= 3696 && (int)blockIdx.x < 5796) && ((((int)blockIdx.x - 3696) / 84) % 1 == 0))
  {
    NewBlockIdx_ = 1848 + (((int)blockIdx.x - 3696) / 84) * 84 + (int)blockIdx.x % 84;
    KernelID_  = 0;
    gridDim_x_ = 125;
    gridDim_y_ = 1;
    gridDim_z_ = 32;
    blockDim_x_ = 8;
    blockDim_y_ = 1;
    blockDim_z_ = 1;
  }
  else if ((int)blockIdx.x >= 5796 && (int)blockIdx.x < 5804)
  {
    NewBlockIdx_ = (int)blockIdx.x - 3948;
    KernelID_  = 1;
    gridDim_x_ = 2;
    gridDim_y_ = 29;
    gridDim_z_ = 32;
    blockDim_x_ = 29;
    blockDim_y_ = 1;
    blockDim_z_ = 2;
  }
  else if ((int)blockIdx.x >= 5804 && (int)blockIdx.x < 5856)
  {
    NewBlockIdx_ = (int)blockIdx.x - 1856;
    KernelID_  = 0;
    gridDim_x_ = 125;
    gridDim_y_ = 1;
    gridDim_z_ = 32;
    blockDim_x_ = 8;
    blockDim_y_ = 1;
    blockDim_z_ = 1;
  }
  blockIdx_x_ = NewBlockIdx_ % gridDim_x_;
  blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;
  blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);
  threadIdx_x_ = (int)threadIdx.x % blockDim_x_;
  threadIdx_y_ = (int)threadIdx.x / blockDim_x_ % blockDim_y_;
  threadIdx_z_ = (int)threadIdx.x / (blockDim_x_ * blockDim_y_);

  static float union_shared_0_[256] __attribute__((shared));
  static float union_shared_1_[232] __attribute__((shared));


  // bgemm
  if ((KernelID_ == 0) && ((int)threadIdx.x < 8))
  {
      float T_batch_matmul_NT_local[1];
      float A_shared_local[1];
      float B_shared_local[1];
      T_batch_matmul_NT_local[0] = 0.F;
      for (int k_outer = 0; k_outer < 64; ++k_outer) {
          __syncthreads();
          union_shared_1_[((int)threadIdx_x_)] = bgemm_A_[(((((int)blockIdx_z_) * 512) + (k_outer * 8)) + ((int)threadIdx_x_))];
          for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
              union_shared_0_[((ax1_inner * 8) + ((int)threadIdx_x_))] = bgemm_B_[(((((((int)blockIdx_z_) * 512000) + (((int)blockIdx_x_) * 4096)) + (ax1_inner * 512)) + (k_outer * 8)) + ((int)threadIdx_x_))];
          }
          __syncthreads();
          for (int k_inner = 0; k_inner < 8; ++k_inner) {
              A_shared_local[0] = union_shared_1_[k_inner];
              B_shared_local[0] = union_shared_0_[((((int)threadIdx_x_) * 8) + k_inner)];
              T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (A_shared_local[0] * B_shared_local[0]));
          }
      }
      bgemm_T_batch_matmul_NT_[(((((int)blockIdx_z_) * 1000) + (((int)blockIdx_x_) * 8)) + ((int)threadIdx_x_))] = T_batch_matmul_NT_local[0];
  }
  // conv2d
  else if ((KernelID_ == 1) && ((int)threadIdx.x < 58))
  {
      float conv2d_nchw_local[64];
      conv2d_nchw_local[0] = 0.F;
      conv2d_nchw_local[2] = 0.F;
      conv2d_nchw_local[4] = 0.F;
      conv2d_nchw_local[6] = 0.F;
      conv2d_nchw_local[8] = 0.F;
      conv2d_nchw_local[10] = 0.F;
      conv2d_nchw_local[12] = 0.F;
      conv2d_nchw_local[14] = 0.F;
      conv2d_nchw_local[16] = 0.F;
      conv2d_nchw_local[18] = 0.F;
      conv2d_nchw_local[20] = 0.F;
      conv2d_nchw_local[22] = 0.F;
      conv2d_nchw_local[24] = 0.F;
      conv2d_nchw_local[26] = 0.F;
      conv2d_nchw_local[28] = 0.F;
      conv2d_nchw_local[30] = 0.F;
      conv2d_nchw_local[32] = 0.F;
      conv2d_nchw_local[34] = 0.F;
      conv2d_nchw_local[36] = 0.F;
      conv2d_nchw_local[38] = 0.F;
      conv2d_nchw_local[40] = 0.F;
      conv2d_nchw_local[42] = 0.F;
      conv2d_nchw_local[44] = 0.F;
      conv2d_nchw_local[46] = 0.F;
      conv2d_nchw_local[48] = 0.F;
      conv2d_nchw_local[50] = 0.F;
      conv2d_nchw_local[52] = 0.F;
      conv2d_nchw_local[54] = 0.F;
      conv2d_nchw_local[56] = 0.F;
      conv2d_nchw_local[58] = 0.F;
      conv2d_nchw_local[60] = 0.F;
      conv2d_nchw_local[62] = 0.F;
      conv2d_nchw_local[1] = 0.F;
      conv2d_nchw_local[3] = 0.F;
      conv2d_nchw_local[5] = 0.F;
      conv2d_nchw_local[7] = 0.F;
      conv2d_nchw_local[9] = 0.F;
      conv2d_nchw_local[11] = 0.F;
      conv2d_nchw_local[13] = 0.F;
      conv2d_nchw_local[15] = 0.F;
      conv2d_nchw_local[17] = 0.F;
      conv2d_nchw_local[19] = 0.F;
      conv2d_nchw_local[21] = 0.F;
      conv2d_nchw_local[23] = 0.F;
      conv2d_nchw_local[25] = 0.F;
      conv2d_nchw_local[27] = 0.F;
      conv2d_nchw_local[29] = 0.F;
      conv2d_nchw_local[31] = 0.F;
      conv2d_nchw_local[33] = 0.F;
      conv2d_nchw_local[35] = 0.F;
      conv2d_nchw_local[37] = 0.F;
      conv2d_nchw_local[39] = 0.F;
      conv2d_nchw_local[41] = 0.F;
      conv2d_nchw_local[43] = 0.F;
      conv2d_nchw_local[45] = 0.F;
      conv2d_nchw_local[47] = 0.F;
      conv2d_nchw_local[49] = 0.F;
      conv2d_nchw_local[51] = 0.F;
      conv2d_nchw_local[53] = 0.F;
      conv2d_nchw_local[55] = 0.F;
      conv2d_nchw_local[57] = 0.F;
      conv2d_nchw_local[59] = 0.F;
      conv2d_nchw_local[61] = 0.F;
      conv2d_nchw_local[63] = 0.F;
      for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
          __syncthreads();
          union_shared_1_[((((int)threadIdx_z_) * 116) + (((int)threadIdx_x_) * 4))] = (((((1 <= ((((int)blockIdx_y_) * 2) + (((((int)threadIdx_x_) * 4) % 58) / 29))) && (((((int)blockIdx_y_) * 2) + (((((int)threadIdx_x_) * 4) % 58) / 29)) < 57)) && (1 <= ((((int)blockIdx_x_) * 29) + ((((int)threadIdx_x_) * 4) % 29)))) && (((((int)blockIdx_x_) * 29) + ((((int)threadIdx_x_) * 4) % 29)) < 57)) ? conv2d_data_[(((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + (((((int)threadIdx_x_) * 2) / 29) * 3136)) + (((int)blockIdx_y_) * 112)) + ((((((int)threadIdx_x_) * 4) % 58) / 29) * 56)) + (((int)blockIdx_x_) * 29)) + ((((int)threadIdx_x_) * 4) % 29)) - 57)] : 0.F);
          union_shared_1_[(((((int)threadIdx_z_) * 116) + (((int)threadIdx_x_) * 4)) + 1)] = (((((1 <= ((((int)blockIdx_y_) * 2) + ((((((int)threadIdx_x_) * 4) + 1) % 58) / 29))) && (((((int)blockIdx_y_) * 2) + ((((((int)threadIdx_x_) * 4) + 1) % 58) / 29)) < 57)) && (1 <= ((((int)blockIdx_x_) * 29) + (((((int)threadIdx_x_) * 4) + 1) % 29)))) && (((((int)blockIdx_x_) * 29) + (((((int)threadIdx_x_) * 4) + 1) % 29)) < 57)) ? conv2d_data_[(((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + (((((int)threadIdx_x_) * 2) / 29) * 3136)) + (((int)blockIdx_y_) * 112)) + (((((((int)threadIdx_x_) * 4) + 1) % 58) / 29) * 56)) + (((int)blockIdx_x_) * 29)) + (((((int)threadIdx_x_) * 4) + 1) % 29)) - 57)] : 0.F);
          union_shared_1_[(((((int)threadIdx_z_) * 116) + (((int)threadIdx_x_) * 4)) + 2)] = (((((1 <= ((((int)blockIdx_y_) * 2) + ((((((int)threadIdx_x_) * 4) + 2) % 58) / 29))) && (((((int)blockIdx_y_) * 2) + ((((((int)threadIdx_x_) * 4) + 2) % 58) / 29)) < 57)) && (1 <= ((((int)blockIdx_x_) * 29) + (((((int)threadIdx_x_) * 4) + 2) % 29)))) && (((((int)blockIdx_x_) * 29) + (((((int)threadIdx_x_) * 4) + 2) % 29)) < 57)) ? conv2d_data_[(((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((((int)threadIdx_x_) * 2) + 1) / 29) * 3136)) + (((int)blockIdx_y_) * 112)) + (((((((int)threadIdx_x_) * 4) + 2) % 58) / 29) * 56)) + (((int)blockIdx_x_) * 29)) + (((((int)threadIdx_x_) * 4) + 2) % 29)) - 57)] : 0.F);
          union_shared_1_[(((((int)threadIdx_z_) * 116) + (((int)threadIdx_x_) * 4)) + 3)] = (((((1 <= ((((int)blockIdx_y_) * 2) + ((((((int)threadIdx_x_) * 4) + 3) % 58) / 29))) && (((((int)blockIdx_y_) * 2) + ((((((int)threadIdx_x_) * 4) + 3) % 58) / 29)) < 57)) && (1 <= ((((int)blockIdx_x_) * 29) + (((((int)threadIdx_x_) * 4) + 3) % 29)))) && (((((int)blockIdx_x_) * 29) + (((((int)threadIdx_x_) * 4) + 3) % 29)) < 57)) ? conv2d_data_[(((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((((int)threadIdx_x_) * 2) + 1) / 29) * 3136)) + (((int)blockIdx_y_) * 112)) + (((((((int)threadIdx_x_) * 4) + 3) % 58) / 29) * 56)) + (((int)blockIdx_x_) * 29)) + (((((int)threadIdx_x_) * 4) + 3) % 29)) - 57)] : 0.F);
          if ((((((int)threadIdx_x_) * 5) >> 7) + ((int)threadIdx_z_)) < 2) {
              if (((int)threadIdx_x_) < 26) {
                  union_shared_0_[((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 5))] = conv2d_kernel_[((((((int)threadIdx_z_) * 2048) + (((((int)threadIdx_x_) * 5) >> 2) * 64)) + (rc_outer * 4)) + (((int)threadIdx_x_) & 3))];
              }
          }
          if (((((((int)threadIdx_x_) * 5) + 1) >> 7) + ((int)threadIdx_z_)) < 2) {
              if (((int)threadIdx_x_) < 26) {
                  union_shared_0_[(((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 5)) + 1)] = conv2d_kernel_[((((((int)threadIdx_z_) * 2048) + ((((((int)threadIdx_x_) * 5) + 1) >> 2) * 64)) + (rc_outer * 4)) + ((((int)threadIdx_x_) + 1) & 3))];
              }
          }
          if (((((((int)threadIdx_x_) * 5) + 2) >> 7) + ((int)threadIdx_z_)) < 2) {
              if (((int)threadIdx_x_) < 26) {
                  union_shared_0_[(((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 5)) + 2)] = conv2d_kernel_[((((((int)threadIdx_z_) * 2048) + ((((((int)threadIdx_x_) * 5) + 2) >> 2) * 64)) + (rc_outer * 4)) + ((((int)threadIdx_x_) + 2) & 3))];
              }
          }
          if (((((((int)threadIdx_x_) * 5) + 3) >> 7) + ((int)threadIdx_z_)) < 2) {
              if (((int)threadIdx_x_) < 25) {
                  union_shared_0_[(((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 5)) + 3)] = conv2d_kernel_[((((((int)threadIdx_z_) * 2048) + ((((((int)threadIdx_x_) * 5) + 3) >> 2) * 64)) + (rc_outer * 4)) + ((((int)threadIdx_x_) + 3) & 3))];
              }
          }
          if (((((((int)threadIdx_x_) * 5) + 4) >> 7) + ((int)threadIdx_z_)) < 2) {
              if (((int)threadIdx_x_) < 25) {
                  union_shared_0_[(((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 5)) + 4)] = conv2d_kernel_[(((((((int)threadIdx_z_) * 2048) + (((((int)threadIdx_x_) * 5) >> 2) * 64)) + (rc_outer * 4)) + (((int)threadIdx_x_) & 3)) + 64)];
              }
          }
          __syncthreads();
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 8)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 16)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 24)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 32)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 40)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 48)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 56)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 72)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 80)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 88)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 96)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 104)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 112)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 120)]));
          conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 128)]));
          conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 136)]));
          conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 144)]));
          conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 152)]));
          conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 160)]));
          conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 168)]));
          conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 176)]));
          conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 184)]));
          conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 192)]));
          conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 200)]));
          conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 208)]));
          conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 216)]));
          conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 224)]));
          conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 232)]));
          conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 240)]));
          conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_1_[((int)threadIdx_x_)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 248)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 8)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 16)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 24)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 32)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 40)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 48)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 56)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 72)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 80)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 88)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 96)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 104)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 112)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 120)]));
          conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 128)]));
          conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 136)]));
          conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 144)]));
          conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 152)]));
          conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 160)]));
          conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 168)]));
          conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 176)]));
          conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 184)]));
          conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 192)]));
          conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 200)]));
          conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 208)]));
          conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 216)]));
          conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 224)]));
          conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 232)]));
          conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 240)]));
          conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_1_[(((int)threadIdx_x_) + 29)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 248)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 9)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 17)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 25)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 33)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 41)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 49)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 57)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 73)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 81)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 89)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 97)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 105)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 113)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 121)]));
          conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 129)]));
          conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 137)]));
          conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 145)]));
          conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 153)]));
          conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 161)]));
          conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 169)]));
          conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 177)]));
          conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 185)]));
          conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 193)]));
          conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 201)]));
          conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 209)]));
          conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 217)]));
          conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 225)]));
          conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 233)]));
          conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 241)]));
          conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_1_[(((int)threadIdx_x_) + 58)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 249)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 9)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 17)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 25)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 33)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 41)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 49)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 57)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 73)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 81)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 89)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 97)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 105)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 113)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 121)]));
          conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 129)]));
          conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 137)]));
          conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 145)]));
          conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 153)]));
          conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 161)]));
          conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 169)]));
          conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 177)]));
          conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 185)]));
          conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 193)]));
          conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 201)]));
          conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 209)]));
          conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 217)]));
          conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 225)]));
          conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 233)]));
          conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 241)]));
          conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_1_[(((int)threadIdx_x_) + 87)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 249)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 10)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 18)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 26)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 34)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 42)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 50)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 58)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 74)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 82)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 90)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 98)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 106)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 114)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 122)]));
          conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 130)]));
          conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 138)]));
          conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 146)]));
          conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 154)]));
          conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 162)]));
          conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 170)]));
          conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 178)]));
          conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 186)]));
          conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 194)]));
          conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 202)]));
          conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 210)]));
          conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 218)]));
          conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 226)]));
          conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 234)]));
          conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 242)]));
          conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_1_[(((int)threadIdx_x_) + 116)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 250)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 10)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 18)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 26)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 34)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 42)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 50)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 58)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 74)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 82)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 90)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 98)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 106)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 114)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 122)]));
          conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 130)]));
          conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 138)]));
          conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 146)]));
          conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 154)]));
          conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 162)]));
          conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 170)]));
          conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 178)]));
          conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 186)]));
          conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 194)]));
          conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 202)]));
          conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 210)]));
          conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 218)]));
          conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 226)]));
          conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 234)]));
          conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 242)]));
          conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_1_[(((int)threadIdx_x_) + 145)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 250)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 11)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 19)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 27)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 35)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 43)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 51)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 59)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 75)]));
          conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 83)]));
          conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 91)]));
          conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 99)]));
          conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 107)]));
          conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 115)]));
          conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 123)]));
          conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 131)]));
          conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 139)]));
          conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 147)]));
          conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 155)]));
          conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 163)]));
          conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 171)]));
          conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 179)]));
          conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 187)]));
          conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 195)]));
          conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 203)]));
          conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 211)]));
          conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 219)]));
          conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 227)]));
          conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 235)]));
          conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 243)]));
          conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_1_[(((int)threadIdx_x_) + 174)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 251)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 11)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 19)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 27)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 35)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 43)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 51)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 59)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 75)]));
          conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 83)]));
          conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 91)]));
          conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 99)]));
          conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 107)]));
          conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 115)]));
          conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 123)]));
          conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 131)]));
          conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 139)]));
          conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 147)]));
          conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 155)]));
          conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 163)]));
          conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 171)]));
          conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 179)]));
          conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 187)]));
          conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 195)]));
          conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 203)]));
          conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 211)]));
          conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 219)]));
          conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 227)]));
          conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 235)]));
          conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 243)]));
          conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_1_[(((int)threadIdx_x_) + 203)] * union_shared_0_[((((int)threadIdx_z_) * 4) + 251)]));
      }
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_))] = conv2d_nchw_local[0];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 6728)] = conv2d_nchw_local[2];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 13456)] = conv2d_nchw_local[4];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 20184)] = conv2d_nchw_local[6];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 26912)] = conv2d_nchw_local[8];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 33640)] = conv2d_nchw_local[10];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 40368)] = conv2d_nchw_local[12];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 47096)] = conv2d_nchw_local[14];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 53824)] = conv2d_nchw_local[16];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 60552)] = conv2d_nchw_local[18];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 67280)] = conv2d_nchw_local[20];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 74008)] = conv2d_nchw_local[22];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 80736)] = conv2d_nchw_local[24];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 87464)] = conv2d_nchw_local[26];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 94192)] = conv2d_nchw_local[28];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 100920)] = conv2d_nchw_local[30];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 107648)] = conv2d_nchw_local[32];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 114376)] = conv2d_nchw_local[34];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 121104)] = conv2d_nchw_local[36];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 127832)] = conv2d_nchw_local[38];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 134560)] = conv2d_nchw_local[40];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 141288)] = conv2d_nchw_local[42];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 148016)] = conv2d_nchw_local[44];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 154744)] = conv2d_nchw_local[46];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 161472)] = conv2d_nchw_local[48];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 168200)] = conv2d_nchw_local[50];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 174928)] = conv2d_nchw_local[52];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 181656)] = conv2d_nchw_local[54];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 188384)] = conv2d_nchw_local[56];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 195112)] = conv2d_nchw_local[58];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 201840)] = conv2d_nchw_local[60];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 208568)] = conv2d_nchw_local[62];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 58)] = conv2d_nchw_local[1];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 6786)] = conv2d_nchw_local[3];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 13514)] = conv2d_nchw_local[5];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 20242)] = conv2d_nchw_local[7];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 26970)] = conv2d_nchw_local[9];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 33698)] = conv2d_nchw_local[11];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 40426)] = conv2d_nchw_local[13];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 47154)] = conv2d_nchw_local[15];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 53882)] = conv2d_nchw_local[17];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 60610)] = conv2d_nchw_local[19];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 67338)] = conv2d_nchw_local[21];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 74066)] = conv2d_nchw_local[23];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 80794)] = conv2d_nchw_local[25];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 87522)] = conv2d_nchw_local[27];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 94250)] = conv2d_nchw_local[29];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 100978)] = conv2d_nchw_local[31];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 107706)] = conv2d_nchw_local[33];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 114434)] = conv2d_nchw_local[35];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 121162)] = conv2d_nchw_local[37];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 127890)] = conv2d_nchw_local[39];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 134618)] = conv2d_nchw_local[41];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 141346)] = conv2d_nchw_local[43];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 148074)] = conv2d_nchw_local[45];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 154802)] = conv2d_nchw_local[47];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 161530)] = conv2d_nchw_local[49];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 168258)] = conv2d_nchw_local[51];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 174986)] = conv2d_nchw_local[53];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 181714)] = conv2d_nchw_local[55];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 188442)] = conv2d_nchw_local[57];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 195170)] = conv2d_nchw_local[59];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 201898)] = conv2d_nchw_local[61];
      conv2d_conv2d_nchw_[((((((((int)blockIdx_z_) * 215296) + (((int)threadIdx_z_) * 3364)) + (((int)blockIdx_y_) * 116)) + (((int)blockIdx_x_) * 29)) + ((int)threadIdx_x_)) + 208626)] = conv2d_nchw_local[63];
  }
}
