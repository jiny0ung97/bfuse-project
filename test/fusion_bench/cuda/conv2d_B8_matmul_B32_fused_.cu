
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
extern "C" __global__ __launch_bounds__(128) void conv2d_B8_matmul_B32_fused_(float *__restrict conv2d_B8_data_, float *__restrict conv2d_B8_kernel_, float *__restrict conv2d_B8_conv2d_nhwc_, float *__restrict matmul_B32_data_, float *__restrict matmul_B32_weight_, float *__restrict matmul_B32_T_matmul_NT_)
{
  static float union_shared_0_[3072] __attribute__((shared));
  static float union_shared_1_[512] __attribute__((shared));

  /*
   * KernelID_ means...
   * 0: conv2d_B8
   * 1: matmul_B32
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
    gridDim_x_ = 125;
    Others_    = 84;
    KernelID_  = 1;
  }
  else if (blockIdx.x >= 168 && blockIdx.x < 252)
  {
    gridDim_x_ = 392;
    Others_    = 84;
    KernelID_  = 0;
  }
  else if (blockIdx.x >= 252 && blockIdx.x < 293)
  {
    gridDim_x_ = 125;
    Others_    = 168;
    KernelID_  = 1;
  }
  else if (blockIdx.x >= 293 && blockIdx.x < 517)
  {
    gridDim_x_ = 392;
    Others_    = 125;
    KernelID_  = 0;
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
  // matmul_B32
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
      for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
          asm ("bar.sync 0, 32;");
          *(float4 *)(union_shared_0_ + (((int)threadIdx.x) * 4)) = *(float4 *)(matmul_B32_data_ + ((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 128)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1024));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 256)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2048));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 384)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3072));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 512)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 4096));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 640)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 5120));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 768)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 6144));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 896)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 7168));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 1024)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 8192));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 1152)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 9216));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 1280)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 10240));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 1408)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 11264));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 1536)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 12288));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 1664)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 13312));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 1792)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 14336));
          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 4) + 1920)) = *(float4 *)(matmul_B32_data_ + (((((((int)threadIdx.x) >> 4) * 512) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 15360));
          *(float4 *)(union_shared_1_ + (((int)threadIdx.x) * 4)) = *(float4 *)(matmul_B32_weight_ + ((((((int)blockIdx_x_) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)));
          *(float4 *)(union_shared_1_ + ((((int)threadIdx.x) * 4) + 128)) = *(float4 *)(matmul_B32_weight_ + (((((((int)blockIdx_x_) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1024));
          *(float4 *)(union_shared_1_ + ((((int)threadIdx.x) * 4) + 256)) = *(float4 *)(matmul_B32_weight_ + (((((((int)blockIdx_x_) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2048));
          *(float4 *)(union_shared_1_ + ((((int)threadIdx.x) * 4) + 384)) = *(float4 *)(matmul_B32_weight_ + (((((((int)blockIdx_x_) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3072));
          asm ("bar.sync 0, 32;");
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[((((int)threadIdx.x) >> 1) * 64)] * union_shared_1_[((((int)threadIdx.x) & 1) * 128)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[((((int)threadIdx.x) >> 1) * 64)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 256)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1024)] * union_shared_1_[((((int)threadIdx.x) & 1) * 128)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1024)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 256)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[((((int)threadIdx.x) >> 1) * 64)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 64)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[((((int)threadIdx.x) >> 1) * 64)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 320)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1024)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 64)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1024)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 320)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 1)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 257)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1025)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 1)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1025)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 257)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 65)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 321)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1025)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 65)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1025)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 321)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 2)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 2)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 2)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 258)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1026)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 2)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1026)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 258)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 2)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 66)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 2)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 322)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1026)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 66)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1026)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 322)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 3)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 3)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 3)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 259)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1027)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 3)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1027)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 259)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 3)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 67)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 3)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 323)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1027)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 67)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1027)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 323)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 4)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 4)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 4)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 260)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1028)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 4)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1028)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 260)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 4)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 68)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 4)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 324)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1028)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 68)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1028)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 324)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 5)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 5)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 5)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 261)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1029)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 5)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1029)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 261)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 5)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 69)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 5)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 325)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1029)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 69)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1029)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 325)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 6)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 6)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 6)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 262)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1030)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 6)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1030)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 262)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 6)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 70)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 6)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 326)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1030)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 70)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1030)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 326)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 7)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 7)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 7)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 263)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1031)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 7)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1031)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 263)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 7)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 71)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 7)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 327)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1031)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 71)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1031)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 327)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 8)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 8)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 8)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 264)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1032)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 8)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1032)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 264)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 8)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 72)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 8)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 328)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1032)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 72)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1032)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 328)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 9)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 9)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 9)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 265)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1033)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 9)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1033)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 265)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 9)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 73)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 9)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 329)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1033)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 73)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1033)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 329)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 10)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 10)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 10)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 266)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1034)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 10)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1034)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 266)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 10)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 74)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 10)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 330)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1034)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 74)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1034)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 330)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 11)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 11)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 11)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 267)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1035)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 11)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1035)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 267)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 11)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 75)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 11)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 331)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1035)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 75)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1035)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 331)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 12)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 12)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 12)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 268)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1036)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 12)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1036)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 268)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 12)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 76)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 12)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 332)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1036)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 76)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1036)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 332)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 13)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 13)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 13)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 269)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1037)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 13)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1037)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 269)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 13)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 77)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 13)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 333)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1037)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 77)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1037)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 333)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 14)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 14)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 14)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 270)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1038)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 14)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1038)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 270)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 14)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 78)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 14)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 334)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1038)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 78)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1038)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 334)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 15)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 15)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 15)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 271)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1039)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 15)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1039)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 271)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 15)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 79)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 15)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 335)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1039)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 79)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1039)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 335)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 16)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 16)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 16)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 272)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1040)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 16)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1040)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 272)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 16)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 80)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 16)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 336)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1040)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 80)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1040)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 336)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 17)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 17)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 17)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 273)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1041)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 17)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1041)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 273)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 17)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 81)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 17)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 337)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1041)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 81)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1041)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 337)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 18)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 18)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 18)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 274)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1042)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 18)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1042)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 274)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 18)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 82)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 18)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 338)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1042)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 82)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1042)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 338)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 19)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 19)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 19)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 275)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1043)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 19)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1043)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 275)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 19)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 83)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 19)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 339)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1043)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 83)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1043)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 339)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 20)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 20)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 20)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 276)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1044)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 20)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1044)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 276)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 20)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 84)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 20)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 340)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1044)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 84)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1044)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 340)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 21)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 21)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 21)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 277)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1045)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 21)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1045)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 277)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 21)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 85)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 21)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 341)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1045)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 85)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1045)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 341)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 22)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 22)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 22)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 278)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1046)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 22)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1046)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 278)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 22)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 86)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 22)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 342)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1046)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 86)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1046)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 342)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 23)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 23)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 23)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 279)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1047)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 23)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1047)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 279)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 23)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 87)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 23)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 343)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1047)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 87)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1047)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 343)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 24)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 24)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 24)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 280)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1048)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 24)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1048)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 280)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 24)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 88)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 24)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 344)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1048)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 88)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1048)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 344)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 25)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 25)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 25)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 281)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1049)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 25)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1049)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 281)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 25)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 89)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 25)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 345)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1049)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 89)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1049)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 345)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 26)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 26)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 26)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 282)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1050)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 26)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1050)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 282)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 26)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 90)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 26)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 346)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1050)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 90)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1050)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 346)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 27)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 27)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 27)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 283)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1051)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 27)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1051)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 283)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 27)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 91)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 27)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 347)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1051)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 91)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1051)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 347)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 28)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 28)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 28)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 284)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1052)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 28)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1052)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 284)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 28)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 92)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 28)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 348)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1052)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 92)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1052)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 348)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 29)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 29)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 29)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 285)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1053)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 29)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1053)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 285)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 29)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 93)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 29)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 349)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1053)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 93)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1053)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 349)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 30)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 30)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 30)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 286)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1054)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 30)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1054)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 286)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 30)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 94)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 30)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 350)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1054)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 94)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1054)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 350)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 31)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 31)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 31)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 287)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1055)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 31)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1055)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 287)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 31)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 95)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 31)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 351)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1055)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 95)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1055)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 351)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 32)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 32)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 32)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 288)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1056)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 32)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1056)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 288)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 32)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 96)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 32)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 352)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1056)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 96)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1056)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 352)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 33)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 33)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 33)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 289)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1057)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 33)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1057)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 289)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 33)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 97)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 33)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 353)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1057)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 97)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1057)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 353)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 34)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 34)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 34)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 290)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1058)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 34)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1058)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 290)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 34)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 98)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 34)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 354)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1058)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 98)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1058)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 354)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 35)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 35)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 35)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 291)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1059)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 35)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1059)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 291)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 35)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 99)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 35)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 355)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1059)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 99)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1059)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 355)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 36)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 36)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 36)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 292)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1060)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 36)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1060)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 292)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 36)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 100)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 36)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 356)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1060)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 100)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1060)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 356)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 37)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 37)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 37)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 293)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1061)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 37)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1061)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 293)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 37)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 101)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 37)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 357)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1061)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 101)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1061)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 357)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 38)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 38)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 38)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 294)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1062)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 38)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1062)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 294)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 38)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 102)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 38)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 358)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1062)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 102)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1062)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 358)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 39)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 39)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 39)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 295)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1063)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 39)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1063)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 295)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 39)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 103)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 39)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 359)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1063)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 103)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1063)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 359)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 40)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 40)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 40)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 296)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1064)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 40)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1064)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 296)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 40)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 104)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 40)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 360)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1064)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 104)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1064)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 360)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 41)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 41)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 41)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 297)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1065)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 41)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1065)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 297)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 41)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 105)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 41)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 361)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1065)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 105)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1065)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 361)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 42)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 42)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 42)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 298)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1066)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 42)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1066)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 298)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 42)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 106)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 42)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 362)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1066)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 106)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1066)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 362)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 43)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 43)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 43)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 299)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1067)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 43)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1067)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 299)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 43)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 107)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 43)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 363)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1067)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 107)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1067)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 363)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 44)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 44)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 44)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 300)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1068)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 44)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1068)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 300)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 44)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 108)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 44)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 364)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1068)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 108)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1068)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 364)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 45)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 45)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 45)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 301)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1069)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 45)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1069)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 301)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 45)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 109)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 45)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 365)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1069)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 109)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1069)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 365)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 46)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 46)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 46)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 302)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1070)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 46)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1070)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 302)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 46)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 110)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 46)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 366)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1070)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 110)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1070)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 366)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 47)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 47)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 47)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 303)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1071)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 47)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1071)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 303)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 47)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 111)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 47)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 367)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1071)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 111)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1071)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 367)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 48)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 48)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 48)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 304)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1072)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 48)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1072)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 304)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 48)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 112)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 48)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 368)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1072)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 112)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1072)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 368)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 49)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 49)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 49)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 305)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1073)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 49)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1073)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 305)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 49)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 113)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 49)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 369)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1073)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 113)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1073)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 369)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 50)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 50)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 50)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 306)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1074)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 50)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1074)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 306)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 50)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 114)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 50)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 370)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1074)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 114)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1074)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 370)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 51)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 51)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 51)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 307)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1075)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 51)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1075)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 307)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 51)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 115)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 51)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 371)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1075)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 115)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1075)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 371)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 52)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 52)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 52)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 308)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1076)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 52)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1076)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 308)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 52)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 116)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 52)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 372)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1076)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 116)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1076)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 372)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 53)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 53)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 53)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 309)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1077)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 53)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1077)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 309)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 53)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 117)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 53)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 373)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1077)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 117)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1077)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 373)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 54)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 54)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 54)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 310)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1078)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 54)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1078)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 310)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 54)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 118)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 54)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 374)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1078)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 118)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1078)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 374)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 55)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 55)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 55)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 311)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1079)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 55)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1079)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 311)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 55)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 119)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 55)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 375)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1079)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 119)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1079)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 375)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 56)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 56)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 56)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 312)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1080)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 56)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1080)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 312)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 56)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 120)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 56)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 376)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1080)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 120)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1080)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 376)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 57)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 57)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 57)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 313)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1081)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 57)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1081)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 313)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 57)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 121)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 57)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 377)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1081)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 121)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1081)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 377)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 58)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 58)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 58)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 314)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1082)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 58)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1082)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 314)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 58)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 122)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 58)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 378)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1082)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 122)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1082)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 378)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 59)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 59)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 59)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 315)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1083)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 59)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1083)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 315)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 59)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 123)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 59)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 379)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1083)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 123)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1083)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 379)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 60)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 60)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 60)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 316)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1084)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 60)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1084)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 316)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 60)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 124)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 60)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 380)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1084)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 124)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1084)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 380)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 61)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 61)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 61)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 317)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1085)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 61)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1085)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 317)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 61)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 125)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 61)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 381)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1085)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 125)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1085)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 381)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 62)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 62)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 62)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 318)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1086)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 62)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1086)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 318)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 62)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 126)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 62)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 382)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1086)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 126)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1086)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 382)]));
          T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 63)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 63)]));
          T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 63)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 319)]));
          T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1087)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 63)]));
          T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1087)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 319)]));
          T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 63)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 127)]));
          T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 63)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 383)]));
          T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1087)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 127)]));
          T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (union_shared_0_[(((((int)threadIdx.x) >> 1) * 64) + 1087)] * union_shared_1_[(((((int)threadIdx.x) & 1) * 128) + 383)]));
      }
      for (int j_inner = 0; j_inner < 2; ++j_inner) {
          matmul_B32_T_matmul_NT_[(((((((int)threadIdx.x) >> 1) * 1000) + (((int)blockIdx_x_) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner)] = T_matmul_NT_local[j_inner];
          matmul_B32_T_matmul_NT_[((((((((int)threadIdx.x) >> 1) * 1000) + (((int)blockIdx_x_) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner) + 4)] = T_matmul_NT_local[(j_inner + 2)];
          matmul_B32_T_matmul_NT_[((((((((int)threadIdx.x) >> 1) * 1000) + (((int)blockIdx_x_) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner) + 16000)] = T_matmul_NT_local[(j_inner + 4)];
          matmul_B32_T_matmul_NT_[((((((((int)threadIdx.x) >> 1) * 1000) + (((int)blockIdx_x_) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner) + 16004)] = T_matmul_NT_local[(j_inner + 6)];
      }
  }
}
