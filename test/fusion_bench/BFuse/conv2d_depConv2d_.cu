

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

extern "C" __global__ __launch_bounds__(112, 4) void conv2d_depConv2d_fused_kernel_bfuse_idx_0(float *__restrict conv2d_conv2d_nchw_, float *__restrict conv2d_data_, float *__restrict conv2d_kernel_, float *__restrict depConv2d_DepthwiseConv2d_, float *__restrict depConv2d_data_, float *__restrict depConv2d_kernel_)
{
  /*
   * KernelID_ means...
   * 0: conv2d
   * 1: depConv2d
   * Kernel's Thread Blocks are 19152
   */
  int gridDim_x_, gridDim_y_, gridDim_z_;
  int blockIdx_x_, blockIdx_y_, blockIdx_z_;
  int blockDim_x_, blockDim_y_, blockDim_z_;
  int threadIdx_x_, threadIdx_y_, threadIdx_z_;
  int NewBlockIdx_;
  int KernelID_ = -1;
  
//   if (((int)blockIdx.x < 16016) && ((int)blockIdx.x % 1596 / 84 >= 0) && ((int)blockIdx.x % 1596 / 84 < 1))
//   {
//     NewBlockIdx_ = ((int)blockIdx.x / 1596) * 84 + (int)blockIdx.x % 1596 - 0;
//     KernelID_  = 0;
//     gridDim_x_ = 1;
//     gridDim_y_ = 28;
//     gridDim_z_ = 32;
//     blockDim_x_ = 56;
//     blockDim_y_ = 1;
//     blockDim_z_ = 2;
//   }
//   else if (((int)blockIdx.x < 17308) && ((int)blockIdx.x % 1596 / 84 >= 1) && ((int)blockIdx.x % 1596 / 84 < 19))
//   {
//     NewBlockIdx_ = ((int)blockIdx.x / 1596) * 1512 + (int)blockIdx.x % 1596 - 84;
//     KernelID_  = 1;
//     gridDim_x_ = 1;
//     gridDim_y_ = 1;
//     gridDim_z_ = 16384;
//     blockDim_x_ = 28;
//     blockDim_y_ = 4;
//     blockDim_z_ = 1;
//   }
//   blockIdx_x_ = NewBlockIdx_ % gridDim_x_;
//   blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;
//   blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);
//   threadIdx_x_ = (int)threadIdx.x % blockDim_x_;
//   threadIdx_y_ = (int)threadIdx.x / blockDim_x_ % blockDim_y_;
//   threadIdx_z_ = (int)threadIdx.x / (blockDim_x_ * blockDim_y_);

  static float union_shared_0_[3249] __attribute__((shared));
  static float union_shared_1_[256] __attribute__((shared));


  // conv2d
  if (((int)threadIdx.x < 112) && ((int)blockIdx.x < 16016) && ((int)blockIdx.x % 1596 / 84 >= 0) && ((int)blockIdx.x % 1596 / 84 < 1))
  {
      NewBlockIdx_ = ((int)blockIdx.x / 1596) * 84 + (int)blockIdx.x % 1596 - 0;
      KernelID_  = 0;
      gridDim_x_ = 1;
      gridDim_y_ = 28;
      gridDim_z_ = 32;
      blockDim_x_ = 56;
      blockDim_y_ = 1;
      blockDim_z_ = 2;
      blockIdx_x_ = NewBlockIdx_ % gridDim_x_;
      blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;
      blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);
      threadIdx_x_ = (int)threadIdx.x % blockDim_x_;
      threadIdx_y_ = (int)threadIdx.x / blockDim_x_ % blockDim_y_;
      threadIdx_z_ = (int)threadIdx.x / (blockDim_x_ * blockDim_y_);
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
          for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
              __syncthreads();
              union_shared_0_[((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4))] = ((((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) && (1 <= (((int)threadIdx_x_) % 14))) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 57)] : 0.F);
              union_shared_0_[(((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4)) + 1)] = (((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 56)] : 0.F);
              union_shared_0_[(((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4)) + 2)] = (((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 55)] : 0.F);
              union_shared_0_[(((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4)) + 3)] = (((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 54)] : 0.F);
              if ((((((int)threadIdx_x_) * 3) >> 7) + ((int)threadIdx_z_)) < 2) {
                  if (((int)threadIdx_x_) < 43) {
                      union_shared_1_[((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 3))] = conv2d_kernel_[(((((((int)threadIdx_z_) * 18432) + (((((int)threadIdx_x_) * 3) >> 2) * 576)) + (rc_outer * 36)) + (((((int)threadIdx_x_) * 3) & 3) * 9)) + (ry_outer * 3))];
                  }
              }
              if (((((((int)threadIdx_x_) * 3) + 1) >> 7) + ((int)threadIdx_z_)) < 2) {
                  if (((int)threadIdx_x_) < 43) {
                      union_shared_1_[(((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 3)) + 1)] = conv2d_kernel_[(((((((int)threadIdx_z_) * 18432) + ((((((int)threadIdx_x_) * 3) + 1) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx_x_) * 3) + 1) & 3) * 9)) + (ry_outer * 3))];
                  }
              }
              if (((((((int)threadIdx_x_) * 3) + 2) >> 7) + ((int)threadIdx_z_)) < 2) {
                  if (((int)threadIdx_x_) < 42) {
                      union_shared_1_[(((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 3)) + 2)] = conv2d_kernel_[(((((((int)threadIdx_z_) * 18432) + ((((((int)threadIdx_x_) * 3) + 2) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx_x_) * 3) + 2) & 3) * 9)) + (ry_outer * 3))];
                  }
              }
              __syncthreads();
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[(((int)threadIdx_z_) * 4)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 8)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 16)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 24)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 32)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 40)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 48)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 56)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 64)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 72)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 80)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 88)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 96)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 104)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 112)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 120)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 128)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 136)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 144)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 152)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 160)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 168)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 176)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 184)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 192)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 200)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 208)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 216)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 224)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 232)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 240)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 248)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[(((int)threadIdx_z_) * 4)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 8)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 16)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 24)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 32)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 40)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 48)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 56)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 64)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 72)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 80)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 88)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 96)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 104)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 112)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 120)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 128)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 136)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 144)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 152)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 160)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 168)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 176)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 184)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 192)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 200)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 208)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 216)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 224)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 232)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 240)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 248)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 1)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 9)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 17)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 25)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 33)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 41)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 49)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 57)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 65)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 73)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 81)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 89)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 97)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 105)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 113)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 121)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 129)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 137)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 145)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 153)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 161)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 169)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 177)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 185)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 193)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 201)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 209)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 217)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 225)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 233)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 241)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 249)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 1)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 9)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 17)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 25)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 33)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 41)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 49)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 57)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 65)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 73)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 81)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 89)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 97)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 105)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 113)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 121)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 129)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 137)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 145)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 153)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 161)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 169)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 177)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 185)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 193)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 201)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 209)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 217)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 225)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 233)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 241)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 249)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 2)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 10)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 18)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 26)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 34)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 42)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 50)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 58)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 66)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 74)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 82)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 90)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 98)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 106)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 114)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 122)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 130)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 138)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 146)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 154)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 162)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 170)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 178)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 186)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 194)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 202)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 210)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 218)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 226)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 234)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 242)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 250)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 2)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 10)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 18)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 26)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 34)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 42)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 50)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 58)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 66)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 74)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 82)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 90)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 98)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 106)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 114)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 122)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 130)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 138)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 146)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 154)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 162)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 170)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 178)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 186)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 194)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 202)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 210)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 218)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 226)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 234)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 242)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 250)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 3)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 11)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 19)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 27)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 35)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 43)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 51)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 59)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 67)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 75)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 83)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 91)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 99)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 107)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 115)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 123)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 131)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 139)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 147)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 155)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 163)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 171)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 179)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 187)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 195)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 203)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 211)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 219)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 227)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 235)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 243)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 251)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 3)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 11)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 19)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 27)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 35)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 43)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 51)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 59)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 67)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 75)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 83)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 91)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 99)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 107)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 115)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 123)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 131)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 139)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 147)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 155)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 163)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 171)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 179)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 187)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 195)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 203)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 211)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 219)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 227)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 235)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 243)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 251)]));
              __syncthreads();
              union_shared_0_[((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4))] = (((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 56)] : 0.F);
              union_shared_0_[(((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4)) + 1)] = (((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 55)] : 0.F);
              union_shared_0_[(((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4)) + 2)] = (((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 54)] : 0.F);
              union_shared_0_[(((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4)) + 3)] = (((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 53)] : 0.F);
              if ((((((int)threadIdx_x_) * 3) >> 7) + ((int)threadIdx_z_)) < 2) {
                  if (((int)threadIdx_x_) < 43) {
                      union_shared_1_[((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 3))] = conv2d_kernel_[((((((((int)threadIdx_z_) * 18432) + (((((int)threadIdx_x_) * 3) >> 2) * 576)) + (rc_outer * 36)) + (((((int)threadIdx_x_) * 3) & 3) * 9)) + (ry_outer * 3)) + 1)];
                  }
              }
              if (((((((int)threadIdx_x_) * 3) + 1) >> 7) + ((int)threadIdx_z_)) < 2) {
                  if (((int)threadIdx_x_) < 43) {
                      union_shared_1_[(((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 3)) + 1)] = conv2d_kernel_[((((((((int)threadIdx_z_) * 18432) + ((((((int)threadIdx_x_) * 3) + 1) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx_x_) * 3) + 1) & 3) * 9)) + (ry_outer * 3)) + 1)];
                  }
              }
              if (((((((int)threadIdx_x_) * 3) + 2) >> 7) + ((int)threadIdx_z_)) < 2) {
                  if (((int)threadIdx_x_) < 42) {
                      union_shared_1_[(((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 3)) + 2)] = conv2d_kernel_[((((((((int)threadIdx_z_) * 18432) + ((((((int)threadIdx_x_) * 3) + 2) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx_x_) * 3) + 2) & 3) * 9)) + (ry_outer * 3)) + 1)];
                  }
              }
              __syncthreads();
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[(((int)threadIdx_z_) * 4)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 8)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 16)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 24)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 32)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 40)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 48)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 56)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 64)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 72)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 80)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 88)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 96)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 104)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 112)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 120)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 128)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 136)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 144)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 152)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 160)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 168)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 176)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 184)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 192)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 200)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 208)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 216)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 224)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 232)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 240)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 248)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[(((int)threadIdx_z_) * 4)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 8)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 16)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 24)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 32)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 40)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 48)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 56)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 64)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 72)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 80)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 88)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 96)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 104)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 112)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 120)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 128)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 136)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 144)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 152)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 160)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 168)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 176)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 184)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 192)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 200)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 208)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 216)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 224)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 232)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 240)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 248)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 1)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 9)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 17)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 25)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 33)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 41)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 49)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 57)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 65)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 73)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 81)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 89)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 97)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 105)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 113)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 121)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 129)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 137)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 145)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 153)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 161)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 169)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 177)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 185)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 193)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 201)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 209)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 217)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 225)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 233)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 241)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 249)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 1)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 9)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 17)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 25)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 33)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 41)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 49)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 57)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 65)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 73)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 81)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 89)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 97)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 105)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 113)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 121)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 129)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 137)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 145)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 153)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 161)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 169)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 177)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 185)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 193)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 201)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 209)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 217)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 225)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 233)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 241)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 249)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 2)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 10)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 18)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 26)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 34)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 42)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 50)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 58)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 66)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 74)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 82)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 90)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 98)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 106)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 114)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 122)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 130)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 138)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 146)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 154)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 162)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 170)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 178)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 186)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 194)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 202)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 210)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 218)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 226)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 234)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 242)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 250)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 2)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 10)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 18)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 26)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 34)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 42)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 50)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 58)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 66)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 74)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 82)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 90)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 98)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 106)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 114)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 122)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 130)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 138)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 146)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 154)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 162)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 170)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 178)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 186)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 194)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 202)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 210)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 218)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 226)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 234)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 242)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 250)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 3)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 11)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 19)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 27)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 35)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 43)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 51)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 59)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 67)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 75)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 83)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 91)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 99)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 107)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 115)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 123)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 131)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 139)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 147)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 155)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 163)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 171)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 179)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 187)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 195)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 203)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 211)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 219)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 227)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 235)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 243)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 251)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 3)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 11)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 19)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 27)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 35)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 43)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 51)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 59)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 67)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 75)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 83)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 91)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 99)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 107)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 115)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 123)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 131)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 139)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 147)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 155)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 163)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 171)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 179)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 187)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 195)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 203)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 211)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 219)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 227)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 235)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 243)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 251)]));
              __syncthreads();
              union_shared_0_[((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4))] = (((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 55)] : 0.F);
              union_shared_0_[(((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4)) + 1)] = (((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 54)] : 0.F);
              union_shared_0_[(((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4)) + 2)] = (((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 53)] : 0.F);
              union_shared_0_[(((((int)threadIdx_z_) * 224) + (((int)threadIdx_x_) * 4)) + 3)] = ((((1 <= (((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer)) && ((((((int)blockIdx_y_) * 2) + ((((int)threadIdx_x_) % 28) / 14)) + ry_outer) < 57)) && ((((int)threadIdx_x_) % 14) < 13)) ? conv2d_data_[((((((((((int)blockIdx_z_) * 200704) + (rc_outer * 12544)) + (((int)threadIdx_z_) * 6272)) + ((((int)threadIdx_x_) / 28) * 3136)) + (((int)blockIdx_y_) * 112)) + (ry_outer * 56)) + ((((int)threadIdx_x_) % 28) * 4)) - 52)] : 0.F);
              if ((((((int)threadIdx_x_) * 3) >> 7) + ((int)threadIdx_z_)) < 2) {
                  if (((int)threadIdx_x_) < 43) {
                      union_shared_1_[((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 3))] = conv2d_kernel_[((((((((int)threadIdx_z_) * 18432) + (((((int)threadIdx_x_) * 3) >> 2) * 576)) + (rc_outer * 36)) + (((((int)threadIdx_x_) * 3) & 3) * 9)) + (ry_outer * 3)) + 2)];
                  }
              }
              if (((((((int)threadIdx_x_) * 3) + 1) >> 7) + ((int)threadIdx_z_)) < 2) {
                  if (((int)threadIdx_x_) < 43) {
                      union_shared_1_[(((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 3)) + 1)] = conv2d_kernel_[((((((((int)threadIdx_z_) * 18432) + ((((((int)threadIdx_x_) * 3) + 1) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx_x_) * 3) + 1) & 3) * 9)) + (ry_outer * 3)) + 2)];
                  }
              }
              if (((((((int)threadIdx_x_) * 3) + 2) >> 7) + ((int)threadIdx_z_)) < 2) {
                  if (((int)threadIdx_x_) < 42) {
                      union_shared_1_[(((((int)threadIdx_z_) * 128) + (((int)threadIdx_x_) * 3)) + 2)] = conv2d_kernel_[((((((((int)threadIdx_z_) * 18432) + ((((((int)threadIdx_x_) * 3) + 2) >> 2) * 576)) + (rc_outer * 36)) + ((((((int)threadIdx_x_) * 3) + 2) & 3) * 9)) + (ry_outer * 3)) + 2)];
                  }
              }
              __syncthreads();
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[(((int)threadIdx_z_) * 4)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 8)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 16)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 24)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 32)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 40)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 48)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 56)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 64)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 72)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 80)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 88)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 96)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 104)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 112)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 120)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 128)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 136)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 144)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 152)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 160)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 168)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 176)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 184)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 192)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 200)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 208)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 216)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 224)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 232)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 240)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[((int)threadIdx_x_)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 248)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[(((int)threadIdx_z_) * 4)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 8)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 16)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 24)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 32)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 40)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 48)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 56)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 64)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 72)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 80)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 88)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 96)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 104)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 112)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 120)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 128)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 136)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 144)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 152)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 160)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 168)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 176)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 184)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 192)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 200)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 208)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 216)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 224)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 232)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 240)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 56)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 248)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 1)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 9)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 17)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 25)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 33)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 41)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 49)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 57)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 65)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 73)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 81)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 89)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 97)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 105)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 113)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 121)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 129)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 137)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 145)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 153)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 161)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 169)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 177)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 185)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 193)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 201)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 209)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 217)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 225)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 233)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 241)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[(((int)threadIdx_x_) + 112)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 249)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 1)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 9)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 17)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 25)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 33)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 41)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 49)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 57)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 65)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 73)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 81)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 89)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 97)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 105)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 113)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 121)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 129)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 137)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 145)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 153)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 161)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 169)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 177)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 185)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 193)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 201)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 209)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 217)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 225)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 233)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 241)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 168)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 249)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 2)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 10)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 18)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 26)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 34)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 42)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 50)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 58)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 66)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 74)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 82)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 90)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 98)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 106)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 114)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 122)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 130)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 138)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 146)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 154)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 162)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 170)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 178)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 186)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 194)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 202)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 210)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 218)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 226)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 234)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 242)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[(((int)threadIdx_x_) + 224)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 250)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 2)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 10)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 18)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 26)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 34)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 42)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 50)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 58)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 66)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 74)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 82)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 90)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 98)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 106)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 114)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 122)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 130)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 138)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 146)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 154)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 162)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 170)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 178)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 186)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 194)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 202)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 210)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 218)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 226)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 234)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 242)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 280)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 250)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 3)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 11)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 19)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 27)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 35)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 43)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 51)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 59)]));
              conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 67)]));
              conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 75)]));
              conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 83)]));
              conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 91)]));
              conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 99)]));
              conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 107)]));
              conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 115)]));
              conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 123)]));
              conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 131)]));
              conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 139)]));
              conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 147)]));
              conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 155)]));
              conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 163)]));
              conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 171)]));
              conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 179)]));
              conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 187)]));
              conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 195)]));
              conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 203)]));
              conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 211)]));
              conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 219)]));
              conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 227)]));
              conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 235)]));
              conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 243)]));
              conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (union_shared_0_[(((int)threadIdx_x_) + 336)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 251)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 3)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 11)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 19)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 27)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 35)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 43)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 51)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 59)]));
              conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 67)]));
              conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 75)]));
              conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 83)]));
              conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 91)]));
              conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 99)]));
              conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 107)]));
              conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 115)]));
              conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 123)]));
              conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 131)]));
              conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 139)]));
              conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 147)]));
              conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 155)]));
              conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 163)]));
              conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 171)]));
              conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 179)]));
              conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 187)]));
              conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 195)]));
              conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 203)]));
              conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 211)]));
              conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 219)]));
              conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 227)]));
              conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 235)]));
              conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 243)]));
              conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (union_shared_0_[(((int)threadIdx_x_) + 392)] * union_shared_1_[((((int)threadIdx_z_) * 4) + 251)]));
          }
      }
      conv2d_conv2d_nchw_[((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_))] = conv2d_nchw_local[0];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 6272)] = conv2d_nchw_local[2];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 12544)] = conv2d_nchw_local[4];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 18816)] = conv2d_nchw_local[6];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 25088)] = conv2d_nchw_local[8];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 31360)] = conv2d_nchw_local[10];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 37632)] = conv2d_nchw_local[12];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 43904)] = conv2d_nchw_local[14];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 50176)] = conv2d_nchw_local[16];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 56448)] = conv2d_nchw_local[18];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 62720)] = conv2d_nchw_local[20];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 68992)] = conv2d_nchw_local[22];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 75264)] = conv2d_nchw_local[24];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 81536)] = conv2d_nchw_local[26];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 87808)] = conv2d_nchw_local[28];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 94080)] = conv2d_nchw_local[30];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 100352)] = conv2d_nchw_local[32];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 106624)] = conv2d_nchw_local[34];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 112896)] = conv2d_nchw_local[36];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 119168)] = conv2d_nchw_local[38];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 125440)] = conv2d_nchw_local[40];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 131712)] = conv2d_nchw_local[42];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 137984)] = conv2d_nchw_local[44];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 144256)] = conv2d_nchw_local[46];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 150528)] = conv2d_nchw_local[48];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 156800)] = conv2d_nchw_local[50];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 163072)] = conv2d_nchw_local[52];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 169344)] = conv2d_nchw_local[54];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 175616)] = conv2d_nchw_local[56];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 181888)] = conv2d_nchw_local[58];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 188160)] = conv2d_nchw_local[60];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 194432)] = conv2d_nchw_local[62];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 56)] = conv2d_nchw_local[1];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 6328)] = conv2d_nchw_local[3];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 12600)] = conv2d_nchw_local[5];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 18872)] = conv2d_nchw_local[7];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 25144)] = conv2d_nchw_local[9];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 31416)] = conv2d_nchw_local[11];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 37688)] = conv2d_nchw_local[13];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 43960)] = conv2d_nchw_local[15];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 50232)] = conv2d_nchw_local[17];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 56504)] = conv2d_nchw_local[19];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 62776)] = conv2d_nchw_local[21];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 69048)] = conv2d_nchw_local[23];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 75320)] = conv2d_nchw_local[25];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 81592)] = conv2d_nchw_local[27];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 87864)] = conv2d_nchw_local[29];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 94136)] = conv2d_nchw_local[31];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 100408)] = conv2d_nchw_local[33];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 106680)] = conv2d_nchw_local[35];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 112952)] = conv2d_nchw_local[37];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 119224)] = conv2d_nchw_local[39];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 125496)] = conv2d_nchw_local[41];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 131768)] = conv2d_nchw_local[43];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 138040)] = conv2d_nchw_local[45];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 144312)] = conv2d_nchw_local[47];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 150584)] = conv2d_nchw_local[49];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 156856)] = conv2d_nchw_local[51];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 163128)] = conv2d_nchw_local[53];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 169400)] = conv2d_nchw_local[55];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 175672)] = conv2d_nchw_local[57];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 181944)] = conv2d_nchw_local[59];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 188216)] = conv2d_nchw_local[61];
      conv2d_conv2d_nchw_[(((((((int)blockIdx_z_) * 200704) + (((int)threadIdx_z_) * 3136)) + (((int)blockIdx_y_) * 112)) + ((int)threadIdx_x_)) + 194488)] = conv2d_nchw_local[63];
  }
  // depConv2d
  else if (((int)threadIdx.x < 112) && ((int)blockIdx.x < 17308) && ((int)blockIdx.x % 1596 / 84 >= 1) && ((int)blockIdx.x % 1596 / 84 < 19))
  {
      NewBlockIdx_ = ((int)blockIdx.x / 1596) * 1512 + (int)blockIdx.x % 1596 - 84;
      KernelID_  = 1;
      gridDim_x_ = 1;
      gridDim_y_ = 1;
      gridDim_z_ = 16384;
      blockDim_x_ = 28;
      blockDim_y_ = 4;
      blockDim_z_ = 1;
      blockIdx_x_ = NewBlockIdx_ % gridDim_x_;
      blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;
      blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);
      threadIdx_x_ = (int)threadIdx.x % blockDim_x_;
      threadIdx_y_ = (int)threadIdx.x / blockDim_x_ % blockDim_y_;
      threadIdx_z_ = (int)threadIdx.x / (blockDim_x_ * blockDim_y_);
      float PaddedInput_shared_local[45];
      float kernel_shared_local[9];
      float DepthwiseConv2d_local[7];
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer < 30; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer) {
          if ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx_y_) * 28)) + ((int)threadIdx_x_)) < 3249) {
              if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 4) + ((int)threadIdx_y_)) < 117) {
                  union_shared_0_[(((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx_y_) * 28)) + ((int)threadIdx_x_))] = (((57 <= (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx_y_) * 28)) + ((int)threadIdx_x_))) && (1 <= ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx_y_) * 28)) + ((int)threadIdx_x_)) % 57))) ? depConv2d_data_[((((((int)blockIdx_z_) * 3136) + (((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx_y_) * 28)) + ((int)threadIdx_x_)) / 57) * 56)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_outer * 112) + (((int)threadIdx_y_) * 28)) + ((int)threadIdx_x_)) % 57)) - 57)] : 0.F);
              }
          }
      }
      if (((((int)threadIdx_y_) * 28) + ((int)threadIdx_x_)) < 9) {
          if (((int)threadIdx_y_) < 1) {
              union_shared_1_[((((int)threadIdx_y_) * 28) + ((int)threadIdx_x_))] = depConv2d_kernel_[(((((int)threadIdx_y_) * 28) + ((((int)blockIdx_z_) & 127) * 9)) + ((int)threadIdx_x_))];
          }
      }
      __syncthreads();
      for (int ax2 = 0; ax2 < 15; ++ax2) {
          for (int ax3 = 0; ax3 < 3; ++ax3) {
              PaddedInput_shared_local[((ax2 * 3) + ax3)] = union_shared_0_[((((((int)threadIdx_y_) * 798) + (ax2 * 57)) + (((int)threadIdx_x_) * 2)) + ax3)];
          }
      }
      for (int ax2_1 = 0; ax2_1 < 3; ++ax2_1) {
          for (int ax3_1 = 0; ax3_1 < 3; ++ax3_1) {
              kernel_shared_local[((ax2_1 * 3) + ax3_1)] = union_shared_1_[((ax2_1 * 3) + ax3_1)];
          }
      }
      for (int i_c = 0; i_c < 7; ++i_c) {
          DepthwiseConv2d_local[i_c] = 0.F;
          for (int di = 0; di < 3; ++di) {
              for (int dj = 0; dj < 3; ++dj) {
                  DepthwiseConv2d_local[i_c] = (DepthwiseConv2d_local[i_c] + (PaddedInput_shared_local[(((i_c * 6) + (di * 3)) + dj)] * kernel_shared_local[((di * 3) + dj)]));
              }
          }
      }
      for (int i_inner_inner_inner = 0; i_inner_inner_inner < 7; ++i_inner_inner_inner) {
          depConv2d_DepthwiseConv2d_[((((((int)blockIdx_z_) * 784) + (((int)threadIdx_y_) * 196)) + (i_inner_inner_inner * 28)) + ((int)threadIdx_x_))] = DepthwiseConv2d_local[i_inner_inner_inner];
      }
  }
}
