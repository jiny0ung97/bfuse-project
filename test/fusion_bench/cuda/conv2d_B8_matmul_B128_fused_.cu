
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
extern "C" __global__ __launch_bounds__(128) void conv2d_B8_matmul_B128_fused_(float *__restrict conv2d_B8_data_, float *__restrict conv2d_B8_kernel_, float *__restrict conv2d_B8_conv2d_nhwc_, float *__restrict matmul_B128_data_, float *__restrict matmul_B128_weight_, float *__restrict matmul_B128_T_matmul_NT_)
{
  static float union_shared_0_[3072] __attribute__((shared));
  static float union_shared_1_[480] __attribute__((shared));

  /*
   * KernelID_ means...
   * 0: conv2d_B8
   * 1: matmul_B128
   */
  int gridDim_x_;
  int blockIdx_x_;
  int Others_;
  int KernelID_;
  
  if (blockIdx.x >= 0 && blockIdx.x < 84)
  {
    gridDim_x_ = 392;
    Others_    = 0;
    KernelID_  = 0;
  }
  else if (blockIdx.x >= 84 && blockIdx.x < 168)
  {
    gridDim_x_ = 500;
    Others_    = 84;
    KernelID_  = 1;
  }
  else if (blockIdx.x >= 168 && blockIdx.x < 252)
  {
    gridDim_x_ = 392;
    Others_    = 84;
    KernelID_  = 0;
  }
  else if (blockIdx.x >= 252 && blockIdx.x < 336)
  {
    gridDim_x_ = 500;
    Others_    = 168;
    KernelID_  = 1;
  }
  else if (blockIdx.x >= 336 && blockIdx.x < 420)
  {
    gridDim_x_ = 392;
    Others_    = 168;
    KernelID_  = 0;
  }
  else if (blockIdx.x >= 420 && blockIdx.x < 504)
  {
    gridDim_x_ = 500;
    Others_    = 252;
    KernelID_  = 1;
  }
  else if (blockIdx.x >= 504 && blockIdx.x < 588)
  {
    gridDim_x_ = 392;
    Others_    = 252;
    KernelID_  = 0;
  }
  else if (blockIdx.x >= 588 && blockIdx.x < 672)
  {
    gridDim_x_ = 500;
    Others_    = 336;
    KernelID_  = 1;
  }
  else if (blockIdx.x >= 672 && blockIdx.x < 728)
  {
    gridDim_x_ = 392;
    Others_    = 336;
    KernelID_  = 0;
  }
  else if (blockIdx.x >= 728 && blockIdx.x < 892)
  {
    gridDim_x_ = 500;
    Others_    = 392;
    KernelID_  = 1;
  }
  blockIdx_x_ = blockIdx.x - Others_;
  
  // conv2d_B8
  if ((KernelID_ == 0) && ((threadIdx.x >= 0 && threadIdx.x < 128)))
  {
      float conv2d_nhwc_local[16];
      for (int nn_c_outer_inner_init = 0; nn_c_outer_inner_init < 2; ++nn_c_outer_inner_init) {
          for (int ff_c_outer_inner_init = 0; ff_c_outer_inner_init < 2; ++ff_c_outer_inner_init) {
              for (int nn_c_inner_init = 0; nn_c_inner_init < 2; ++nn_c_inner_init) {
                  conv2d_nhwc_local[(((nn_c_outer_inner_init * 4) + (nn_c_inner_init * 2)) + ff_c_outer_inner_init)] = 0.F;
                  conv2d_nhwc_local[((((nn_c_outer_inner_init * 4) + (nn_c_inner_init * 2)) + ff_c_outer_inner_init) + 8)] = 0.F;
              }
          }
      }
      for (int ry_outer_outer = 0; ry_outer_outer < 3; ++ry_outer_outer) {
          for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
              asm ("bar.sync 0, 128;");
              if (((int)threadIdx.x) < 120) {
                  *(float4 *)(union_shared_1_ + (((int)threadIdx.x) * 4)) = (((1 <= (((((((int)blockIdx_x_) % 196) / 14) * 4) + ((((int)threadIdx.x) % 30) / 10)) + ry_outer_outer)) && (1 <= (((((int)blockIdx_x_) % 14) * 4) + ((((int)threadIdx.x) % 10) >> 1)))) ? *(float4 *)(conv2d_B8_data_ + (((((((((((((int)blockIdx_x_) / 196) * 802816) + ((((int)threadIdx.x) / 30) * 200704)) + (((((int)blockIdx_x_) % 196) / 14) * 14336)) + (((((int)threadIdx.x) % 30) / 10) * 3584)) + (ry_outer_outer * 3584)) + ((((int)blockIdx_x_) % 14) * 256)) + (((((int)threadIdx.x) % 10) >> 1) * 64)) + (rc_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) - 3648)) : make_float4(0.F, 0.F, 0.F, 0.F));
              }
              for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 6; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
                  *(float4 *)(union_shared_0_ + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 512) + (((int)threadIdx.x) * 4))) = *(float4 *)(conv2d_B8_kernel_ + (((((ry_outer_outer * 24576) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer >> 1) * 8192)) + (rc_outer_outer * 1024)) + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer & 1) * 512)) + (((int)threadIdx.x) * 4)));
              }
              asm ("bar.sync 0, 128;");
              for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
                  for (int nn_c_outer_inner = 0; nn_c_outer_inner < 2; ++nn_c_outer_inner) {
                      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 2; ++ff_c_outer_inner) {
                          for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
                              for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
                                  for (int nn_c_inner = 0; nn_c_inner < 2; ++nn_c_inner) {
                                      conv2d_nhwc_local[(((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner)] = (conv2d_nhwc_local[(((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner)] + (union_shared_1_[((((((nn_c_outer_inner * 240) + (nn_c_inner * 120)) + ((((int)threadIdx.x) >> 6) * 80)) + (rx_inner * 8)) + (rc_outer_inner * 2)) + rc_inner)] * union_shared_0_[(((((rx_inner * 1024) + (rc_outer_inner * 256)) + (rc_inner * 128)) + ((((int)threadIdx.x) & 63) * 2)) + ff_c_outer_inner)]));
                                      conv2d_nhwc_local[((((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner) + 8)] = (conv2d_nhwc_local[((((nn_c_outer_inner * 4) + (nn_c_inner * 2)) + ff_c_outer_inner) + 8)] + (union_shared_1_[(((((((nn_c_outer_inner * 240) + (nn_c_inner * 120)) + ((((int)threadIdx.x) >> 6) * 80)) + (rx_inner * 8)) + (rc_outer_inner * 2)) + rc_inner) + 16)] * union_shared_0_[(((((rx_inner * 1024) + (rc_outer_inner * 256)) + (rc_inner * 128)) + ((((int)threadIdx.x) & 63) * 2)) + ff_c_outer_inner)]));
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
              conv2d_B8_conv2d_nhwc_[((((((((((int)blockIdx_x_) / 196) * 401408) + (nn_inner * 100352)) + (((((int)blockIdx_x_) % 196) / 14) * 7168)) + ((((int)threadIdx.x) >> 6) * 3584)) + ((((int)blockIdx_x_) % 14) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + ff_inner)] = conv2d_nhwc_local[((nn_inner * 2) + ff_inner)];
              conv2d_B8_conv2d_nhwc_[(((((((((((int)blockIdx_x_) / 196) * 401408) + (nn_inner * 100352)) + (((((int)blockIdx_x_) % 196) / 14) * 7168)) + ((((int)threadIdx.x) >> 6) * 3584)) + ((((int)blockIdx_x_) % 14) * 256)) + ((((int)threadIdx.x) & 63) * 2)) + ff_inner) + 128)] = conv2d_nhwc_local[(((nn_inner * 2) + ff_inner) + 8)];
          }
      }
  }
  // matmul_B128
  else if ((KernelID_ == 1) && ((threadIdx.x >= 0 && threadIdx.x < 32)))
  {
      float T_matmul_NT_local[8];
      T_matmul_NT_local[0] = 0.F;
      T_matmul_NT_local[2] = 0.F;
      T_matmul_NT_local[4] = 0.F;
      T_matmul_NT_local[6] = 0.F;
      T_matmul_NT_local[1] = 0.F;
      T_matmul_NT_local[3] = 0.F;
      T_matmul_NT_local[5] = 0.F;
      T_matmul_NT_local[7] = 0.F;
      for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
          asm ("bar.sync 0, 32;");
          *(float4 *)(union_shared_0_ + (((int)threadIdx.x) * 4)) = *(float4 *)(matmul_B128_data_ + (((((((int)blockIdx_x_) / 125) * 16384) + ((((int)threadIdx.x) >> 1) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 128)) = *(float4 *)(matmul_B128_data_ + ((((((((int)blockIdx_x_) / 125) * 16384) + ((((int)threadIdx.x) >> 1) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 8192));
          *(float2 *)(union_shared_1_ + (((int)threadIdx.x) * 2)) = *(float2 *)(matmul_B128_weight_ + (((((((int)blockIdx_x_) % 125) * 4096) + ((((int)threadIdx.x) >> 2) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)));
          asm ("bar.sync 0, 32;");
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[((((int)threadIdx.x) >> 1) * 8)] * union_shared_1_[((((int)threadIdx.x) & 1) * 16)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[((((int)threadIdx.x) >> 1) * 8)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 32)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 128)] * union_shared_1_[((((int)threadIdx.x) & 1) * 16)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 128)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 32)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[((((int)threadIdx.x) >> 1) * 8)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 8)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[((((int)threadIdx.x) >> 1) * 8)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 40)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 128)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 8)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 128)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 40)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 1)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 1)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 1)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 33)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 129)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 1)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 129)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 33)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 1)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 9)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 1)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 41)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 129)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 9)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 129)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 41)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 2)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 2)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 2)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 34)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 130)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 2)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 130)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 34)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 2)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 10)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 2)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 42)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 130)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 10)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 130)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 42)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 3)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 3)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 3)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 35)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 131)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 3)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 131)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 35)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 3)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 11)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 3)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 43)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 131)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 11)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 131)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 43)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 4)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 4)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 4)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 36)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 132)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 4)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 132)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 36)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 4)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 12)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 4)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 44)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 132)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 12)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 132)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 44)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 5)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 5)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 5)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 37)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 133)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 5)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 133)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 37)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 5)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 13)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 5)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 45)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 133)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 13)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 133)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 45)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 6)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 6)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 6)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 38)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 134)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 6)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 134)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 38)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 6)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 14)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 6)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 46)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 134)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 14)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 134)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 46)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 7)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 7)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 7)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 39)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 135)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 7)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 135)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 39)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 7)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 15)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 7)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 47)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 135)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 15)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 8) + 135)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 16) + 47)]));
      }
      for (int j_inner = 0; j_inner < 2; ++j_inner) {
          matmul_B128_T_matmul_NT_[((((((((int)blockIdx_x_) / 125) * 32000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx_x_) % 125) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner)] = T_matmul_NT_local[j_inner];
          matmul_B128_T_matmul_NT_[(((((((((int)blockIdx_x_) / 125) * 32000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx_x_) % 125) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner) + 4)] = T_matmul_NT_local[(j_inner + 2)];
          matmul_B128_T_matmul_NT_[(((((((((int)blockIdx_x_) / 125) * 32000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx_x_) % 125) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner) + 16000)] = T_matmul_NT_local[(j_inner + 4)];
          matmul_B128_T_matmul_NT_[(((((((((int)blockIdx_x_) / 125) * 32000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx_x_) % 125) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner) + 16004)] = T_matmul_NT_local[(j_inner + 6)];
      }
  }
}
