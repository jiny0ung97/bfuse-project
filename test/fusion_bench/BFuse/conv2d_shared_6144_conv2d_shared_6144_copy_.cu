

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

extern "C" __global__ __launch_bounds__(256) void conv2d_shared_6144_conv2d_shared_6144_copy_fused_kernel_bfuse_idx_0(float *__restrict conv2d_shared_6144_A_, float *__restrict conv2d_shared_6144_B_, float *__restrict conv2d_shared_6144_W_, float *__restrict conv2d_shared_6144_copy_A_, float *__restrict conv2d_shared_6144_copy_B_, float *__restrict conv2d_shared_6144_copy_W_)
{
  /*
   * KernelID_ means...
   * 0: conv2d_shared_6144
   * 1: conv2d_shared_6144_copy
   */
  int gridDim_x_, gridDim_y_, gridDim_z_;
  int blockIdx_x_, blockIdx_y_, blockIdx_z_;
  int TotalBlockIdx_;
  int KernelID_;
  
  if (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y >= 0 && blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y < 1568)
  {
    TotalBlockIdx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y - 0;
    KernelID_  = 0;
    gridDim_x_ = 1568;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
  }
  else if (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y >= 1568 && blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y < 3136)
  {
    TotalBlockIdx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y - 1568;
    KernelID_  = 1;
    gridDim_x_ = 1568;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
  }
  blockIdx_x_ = TotalBlockIdx_ % gridDim_x_;
  blockIdx_y_ = TotalBlockIdx_ / gridDim_x_ % gridDim_y_;
  blockIdx_z_ = TotalBlockIdx_ / (gridDim_x_ * gridDim_y_);

  static float union_shared_0_[3072] __attribute__((shared));
  static float union_shared_1_[3072] __attribute__((shared));


  // conv2d_shared_6144
  if ((KernelID_ == 0) && ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y >= 0 && threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y < 256)))
  {
      float B_local[64];
      float Apad_shared_local[8];
      float W_shared_local[8];
      for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
          for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
              B_local[((ff_c_init * 4) + nn_c_init)] = 0.F;
              B_local[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
              B_local[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
              B_local[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
          }
      }
      for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
          for (int ry = 0; ry < 3; ++ry) {
              for (int rx = 0; rx < 3; ++rx) {
                  asm ("bar.sync 0, 256;");
                  for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                      if (((int)threadIdx.x) < 128) {
                          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx_x_) / 112) + ry)) && (((((int)blockIdx_x_) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx_x_) % 112) >> 3) + rx))) && ((((((int)blockIdx_x_) % 112) >> 3) + rx) < 15)) ? *(float4 *)(conv2d_shared_6144_A_ + (((((((((ry * 917504) + ((((int)blockIdx_x_) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx_x_) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                      }
                  }
                  for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                      if (((int)threadIdx.x) < 128) {
                          *(float4 *)(union_shared_1_ + ((((int)threadIdx.x) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(conv2d_shared_6144_W_ + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx.x) >> 4) * 512)) + (((((int)blockIdx_x_) & 7) >> 1) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                      }
                  }
                  asm ("bar.sync 0, 256;");
                  for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                      for (int ax3 = 0; ax3 < 4; ++ax3) {
                          Apad_shared_local[ax3] = union_shared_0_[(((rc_inner * 128) + ((((int)threadIdx.x) & 15) * 4)) + ax3)];
                          Apad_shared_local[(ax3 + 4)] = union_shared_0_[((((rc_inner * 128) + ((((int)threadIdx.x) & 15) * 4)) + ax3) + 64)];
                      }
                      for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                          W_shared_local[ax3_1] = union_shared_1_[(((rc_inner * 128) + ((((int)threadIdx.x) >> 4) * 4)) + ax3_1)];
                          W_shared_local[(ax3_1 + 4)] = union_shared_1_[((((rc_inner * 128) + ((((int)threadIdx.x) >> 4) * 4)) + ax3_1) + 64)];
                      }
                      for (int ff_c = 0; ff_c < 4; ++ff_c) {
                          for (int nn_c = 0; nn_c < 4; ++nn_c) {
                              B_local[((ff_c * 4) + nn_c)] = (B_local[((ff_c * 4) + nn_c)] + (Apad_shared_local[nn_c] * W_shared_local[ff_c]));
                              B_local[(((ff_c * 4) + nn_c) + 32)] = (B_local[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local[nn_c] * W_shared_local[(ff_c + 4)]));
                              B_local[(((ff_c * 4) + nn_c) + 16)] = (B_local[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local[(nn_c + 4)] * W_shared_local[ff_c]));
                              B_local[(((ff_c * 4) + nn_c) + 48)] = (B_local[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local[(nn_c + 4)] * W_shared_local[(ff_c + 4)]));
                          }
                      }
                  }
              }
          }
      }
      for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
          for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
              conv2d_shared_6144_B_[(((((((((int)blockIdx_x_) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx_x_) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner)] = B_local[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
              conv2d_shared_6144_B_[((((((((((int)blockIdx_x_) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx_x_) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
              conv2d_shared_6144_B_[((((((((((int)blockIdx_x_) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx_x_) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
              conv2d_shared_6144_B_[((((((((((int)blockIdx_x_) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx_x_) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
          }
      }
  }
  // conv2d_shared_6144_copy
  else if ((KernelID_ == 1) && ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y >= 0 && threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y < 256)))
  {
      float B_local[64];
      float Apad_shared_local[8];
      float W_shared_local[8];
      for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
          for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
              B_local[((ff_c_init * 4) + nn_c_init)] = 0.F;
              B_local[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.F;
              B_local[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.F;
              B_local[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.F;
          }
      }
      for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
          for (int ry = 0; ry < 3; ++ry) {
              for (int rx = 0; rx < 3; ++rx) {
                  asm ("bar.sync 0, 256;");
                  for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                      if (((int)threadIdx.x) < 128) {
                          *(float4 *)(union_shared_0_ + ((((int)threadIdx.x) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx_x_) / 112) + ry)) && (((((int)blockIdx_x_) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx_x_) % 112) >> 3) + rx))) && ((((((int)blockIdx_x_) % 112) >> 3) + rx) < 15)) ? *(float4 *)(conv2d_shared_6144_copy_A_ + (((((((((ry * 917504) + ((((int)blockIdx_x_) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx_x_) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.F, 0.F, 0.F, 0.F));
                      }
                  }
                  for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                      if (((int)threadIdx.x) < 128) {
                          *(float4 *)(union_shared_1_ + ((((int)threadIdx.x) * 8) + (ax3_inner_outer_1 * 4))) = *(float4 *)(conv2d_shared_6144_copy_W_ + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx.x) >> 4) * 512)) + (((((int)blockIdx_x_) & 7) >> 1) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
                      }
                  }
                  asm ("bar.sync 0, 256;");
                  for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                      for (int ax3 = 0; ax3 < 4; ++ax3) {
                          Apad_shared_local[ax3] = union_shared_0_[(((rc_inner * 128) + ((((int)threadIdx.x) & 15) * 4)) + ax3)];
                          Apad_shared_local[(ax3 + 4)] = union_shared_0_[((((rc_inner * 128) + ((((int)threadIdx.x) & 15) * 4)) + ax3) + 64)];
                      }
                      for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                          W_shared_local[ax3_1] = union_shared_1_[(((rc_inner * 128) + ((((int)threadIdx.x) >> 4) * 4)) + ax3_1)];
                          W_shared_local[(ax3_1 + 4)] = union_shared_1_[((((rc_inner * 128) + ((((int)threadIdx.x) >> 4) * 4)) + ax3_1) + 64)];
                      }
                      for (int ff_c = 0; ff_c < 4; ++ff_c) {
                          for (int nn_c = 0; nn_c < 4; ++nn_c) {
                              B_local[((ff_c * 4) + nn_c)] = (B_local[((ff_c * 4) + nn_c)] + (Apad_shared_local[nn_c] * W_shared_local[ff_c]));
                              B_local[(((ff_c * 4) + nn_c) + 32)] = (B_local[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local[nn_c] * W_shared_local[(ff_c + 4)]));
                              B_local[(((ff_c * 4) + nn_c) + 16)] = (B_local[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local[(nn_c + 4)] * W_shared_local[ff_c]));
                              B_local[(((ff_c * 4) + nn_c) + 48)] = (B_local[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local[(nn_c + 4)] * W_shared_local[(ff_c + 4)]));
                          }
                      }
                  }
              }
          }
      }
      for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
          for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
              conv2d_shared_6144_copy_B_[(((((((((int)blockIdx_x_) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx_x_) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner)] = B_local[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
              conv2d_shared_6144_copy_B_[((((((((((int)blockIdx_x_) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx_x_) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
              conv2d_shared_6144_copy_B_[((((((((((int)blockIdx_x_) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx_x_) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
              conv2d_shared_6144_copy_B_[((((((((((int)blockIdx_x_) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx_x_) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
          }
      }
  }
}
