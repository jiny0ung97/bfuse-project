

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

extern "C" __global__ __launch_bounds__(256) void conv2d_small_conv2d_large_fused_kernel_bfuse_idx_0(float *__restrict conv2d_small_A_, float *__restrict conv2d_small_B_, float *__restrict conv2d_small_W_, float *__restrict conv2d_large_A_, float *__restrict conv2d_large_B_, float *__restrict conv2d_large_W_)
{
  /*
   * KernelID_ means...
   * 0: conv2d_small
   * 1: conv2d_large
   */
  int gridDim_x_, gridDim_y_, gridDim_z_;
  int blockIdx_x_, blockIdx_y_, blockIdx_z_;
  int blockDim_x_, blockDim_y_, blockDim_z_;
  int threadIdx_x_, threadIdx_y_, threadIdx_z_;
  int NewBlockIdx_;
  int KernelID_;
  
  if (((int)blockIdx.x >= 0 && (int)blockIdx.x < 6216) && ((((int)blockIdx.x - 0) / 84) % 2 == 0))
  {
    NewBlockIdx_ = 0 + (((int)blockIdx.x - 0) / 168) * 84 + (int)blockIdx.x % 84;
    KernelID_  = 1;
    gridDim_x_ = 1;
    gridDim_y_ = 1;
    gridDim_z_ = 3136;
    blockDim_x_ = 8;
    blockDim_y_ = 8;
    blockDim_z_ = 1;
  }
  else if (((int)blockIdx.x >= 0 && (int)blockIdx.x < 6216) && ((((int)blockIdx.x - 0) / 84) % 2 == 1))
  {
    NewBlockIdx_ = 0 + (((int)blockIdx.x - 0) / 168) * 84 + (int)blockIdx.x % 84;
    KernelID_  = 0;
    gridDim_x_ = 1;
    gridDim_y_ = 1;
    gridDim_z_ = 3364;
    blockDim_x_ = 16;
    blockDim_y_ = 16;
    blockDim_z_ = 1;
  }
  else if (((int)blockIdx.x >= 6216 && (int)blockIdx.x < 6468) && ((((int)blockIdx.x - 6216) / 84) % 1 == 0))
  {
    NewBlockIdx_ = 3108 + (((int)blockIdx.x - 6216) / 84) * 84 + (int)blockIdx.x % 84;
    KernelID_  = 0;
    gridDim_x_ = 1;
    gridDim_y_ = 1;
    gridDim_z_ = 3364;
    blockDim_x_ = 16;
    blockDim_y_ = 16;
    blockDim_z_ = 1;
  }
  else if ((int)blockIdx.x >= 6468 && (int)blockIdx.x < 6496)
  {
    NewBlockIdx_ = (int)blockIdx.x - 3360;
    KernelID_  = 1;
    gridDim_x_ = 1;
    gridDim_y_ = 1;
    gridDim_z_ = 3136;
    blockDim_x_ = 8;
    blockDim_y_ = 8;
    blockDim_z_ = 1;
  }
  else if ((int)blockIdx.x >= 6496 && (int)blockIdx.x < 6500)
  {
    NewBlockIdx_ = (int)blockIdx.x - 3136;
    KernelID_  = 0;
    gridDim_x_ = 1;
    gridDim_y_ = 1;
    gridDim_z_ = 3364;
    blockDim_x_ = 16;
    blockDim_y_ = 16;
    blockDim_z_ = 1;
  }
  blockIdx_x_ = NewBlockIdx_ % gridDim_x_;
  blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;
  blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);
  threadIdx_x_ = (int)threadIdx.x % blockDim_x_;
  threadIdx_y_ = (int)threadIdx.x / blockDim_x_ % blockDim_y_;
  threadIdx_z_ = (int)threadIdx.x / (blockDim_x_ * blockDim_y_);

//   static float union_shared_0_[1024] __attribute__((shared));
//   static float union_shared_1_[512] __attribute__((shared));
    static float union_shared_0_[2048] __attribute__((shared));
  static float union_shared_1_[2048] __attribute__((shared));


  // conv2d_small
  if ((KernelID_ == 0) && ((int)threadIdx.x < 256))
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
      for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
          __syncthreads();
          for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
              if (((int)threadIdx_y_) < 8) {
                  *(float4 *)(union_shared_0_ + (((((int)threadIdx_y_) * 128) + (((int)threadIdx_x_) * 8)) + (ax3_inner_outer * 4))) = (((((58 <= ((int)blockIdx_z_)) && (((int)blockIdx_z_) < 3306)) && (1 <= (((int)blockIdx_z_) % 58))) && ((((int)blockIdx_z_) % 58) < 57)) ? *(float4 *)(conv2d_small_A_ + ((((((((((int)blockIdx_z_) / 58) * 458752) + ((((int)blockIdx_z_) % 58) * 8192)) + (rc_outer * 1024)) + (((int)threadIdx_y_) * 128)) + (((int)threadIdx_x_) * 8)) + (ax3_inner_outer * 4)) - 466944)) : make_float4(0.F, 0.F, 0.F, 0.F));
              }
          }
          if (((int)threadIdx_y_) < 8) {
              *(float4 *)(union_shared_1_ + ((((int)threadIdx_y_) * 64) + (((int)threadIdx_x_) * 4))) = *(float4 *)(conv2d_small_W_ + (((rc_outer * 512) + (((int)threadIdx_y_) * 64)) + (((int)threadIdx_x_) * 4)));
          }
          __syncthreads();
          for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
              for (int ax3 = 0; ax3 < 4; ++ax3) {
                  Apad_shared_local[ax3] = union_shared_0_[(((rc_inner * 128) + (((int)threadIdx_x_) * 4)) + ax3)];
                  Apad_shared_local[(ax3 + 4)] = union_shared_0_[((((rc_inner * 128) + (((int)threadIdx_x_) * 4)) + ax3) + 64)];
              }
              for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                  W_shared_local[ax3_1] = union_shared_1_[(((rc_inner * 64) + (((int)threadIdx_y_) * 4)) + ax3_1)];
              }
              for (int ff_c = 0; ff_c < 4; ++ff_c) {
                  for (int nn_c = 0; nn_c < 4; ++nn_c) {
                      B_local[((ff_c * 4) + nn_c)] = (B_local[((ff_c * 4) + nn_c)] + (Apad_shared_local[nn_c] * W_shared_local[ff_c]));
                      B_local[(((ff_c * 4) + nn_c) + 16)] = (B_local[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local[(nn_c + 4)] * W_shared_local[ff_c]));
                  }
              }
          }
      }
      for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
          for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
              conv2d_small_B_[(((((((int)blockIdx_z_) * 8192) + (((int)threadIdx_y_) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_) * 4)) + nn_inner_inner_inner)] = B_local[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
              conv2d_small_B_[((((((((int)blockIdx_z_) * 8192) + (((int)threadIdx_y_) * 512)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx_x_) * 4)) + nn_inner_inner_inner) + 64)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
          }
      }
  }
  // conv2d_large
  else if ((KernelID_ == 1) && ((int)threadIdx.x < 64))
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
      for (int rc_outer = 0; rc_outer < 8; ++rc_outer) {
          for (int ry = 0; ry < 3; ++ry) {
              for (int rx = 0; rx < 3; ++rx) {
                  __syncthreads();
                  for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
                      *(float4 *)(union_shared_0_ + (((((int)threadIdx_y_) * 64) + (((int)threadIdx_x_) * 8)) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx_z_) / 56) + ry)) && (((((int)blockIdx_z_) / 56) + ry) < 57)) && (1 <= (rx + (((int)blockIdx_z_) % 56)))) && ((rx + (((int)blockIdx_z_) % 56)) < 57)) ? *(float4 *)(conv2d_large_A_ + ((((((((ry * 229376) + (((int)blockIdx_z_) * 4096)) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_) * 64)) + (((int)threadIdx_x_) * 8)) + (ax3_inner_outer * 4)) - 233472)) : make_float4(0.F, 0.F, 0.F, 0.F));
                  }
                  for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
                      *(float4 *)(union_shared_1_ + (((((int)threadIdx_y_) * 64) + (((int)threadIdx_x_) * 8)) + (ax3_inner_outer_1 * 4))) = *(float4 *)(conv2d_large_W_ + ((((((ry * 12288) + (rx * 4096)) + (rc_outer * 512)) + (((int)threadIdx_y_) * 64)) + (((int)threadIdx_x_) * 8)) + (ax3_inner_outer_1 * 4)));
                  }
                  __syncthreads();
                  for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
                      for (int ax3 = 0; ax3 < 4; ++ax3) {
                          Apad_shared_local[ax3] = union_shared_0_[(((rc_inner * 64) + (((int)threadIdx_x_) * 4)) + ax3)];
                          Apad_shared_local[(ax3 + 4)] = union_shared_0_[((((rc_inner * 64) + (((int)threadIdx_x_) * 4)) + ax3) + 32)];
                      }
                      for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
                          W_shared_local[ax3_1] = union_shared_1_[(((rc_inner * 64) + (((int)threadIdx_y_) * 4)) + ax3_1)];
                          W_shared_local[(ax3_1 + 4)] = union_shared_1_[((((rc_inner * 64) + (((int)threadIdx_y_) * 4)) + ax3_1) + 32)];
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
              conv2d_large_B_[(((((((int)blockIdx_z_) * 4096) + (((int)threadIdx_y_) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_) * 4)) + nn_inner_inner_inner)] = B_local[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
              conv2d_large_B_[((((((((int)blockIdx_z_) * 4096) + (((int)threadIdx_y_) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_) * 4)) + nn_inner_inner_inner) + 2048)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
              conv2d_large_B_[((((((((int)blockIdx_z_) * 4096) + (((int)threadIdx_y_) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_) * 4)) + nn_inner_inner_inner) + 32)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
              conv2d_large_B_[((((((((int)blockIdx_z_) * 4096) + (((int)threadIdx_y_) * 256)) + (ff_inner_inner_inner * 64)) + (((int)threadIdx_x_) * 4)) + nn_inner_inner_inner) + 2080)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
          }
      }
  }
}
