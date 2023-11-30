

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

extern "C" __global__ __launch_bounds__(128) void softmax_shared_12_softmax_shared_12_copy_fused_kernel_bfuse_idx_0(float *__restrict softmax_shared_12_T_softmax_norm_, float *__restrict softmax_shared_12_data_, float *__restrict softmax_shared_12_copy_T_softmax_norm_, float *__restrict softmax_shared_12_copy_data_)
{
  /*
   * KernelID_ means...
   * 0: softmax_shared_12
   * 1: softmax_shared_12_copy
   */
  int gridDim_x_, gridDim_y_, gridDim_z_;
  int blockIdx_x_, blockIdx_y_, blockIdx_z_;
  int TotalBlockIdx_;
  int KernelID_;
  
  if (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y >= 0 && blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y < 128)
  {
    TotalBlockIdx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y - 0;
    KernelID_  = 0;
    gridDim_x_ = 128;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
  }
  else if (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y >= 128 && blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y < 256)
  {
    TotalBlockIdx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y - 128;
    KernelID_  = 1;
    gridDim_x_ = 128;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
  }
  blockIdx_x_ = TotalBlockIdx_ % gridDim_x_;
  blockIdx_y_ = TotalBlockIdx_ / gridDim_x_ % gridDim_y_;
  blockIdx_z_ = TotalBlockIdx_ / (gridDim_x_ * gridDim_y_);

  static float union_shared_0_[4] __attribute__((shared));
  static float union_shared_1_[4] __attribute__((shared));
  static float union_shared_2_[1] __attribute__((shared));
  static float union_shared_3_[1] __attribute__((shared));
  static float union_shared_4_[1] __attribute__((shared));
  static float union_shared_5_[1] __attribute__((shared));


  // softmax_shared_12
  if ((KernelID_ == 0) && ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y >= 0 && threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y < 128)))
  {
      float normal_reduce_temp0[1];
      float normal_reduce_temp0_1[1];
      normal_reduce_temp0[0] = -3.40282306E+38F;
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_data_[((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x))]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 128)]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 256)]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 384)]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 512)]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 640)]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 768)]);
      if (((int)threadIdx.x) < 104) {
          normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 896)]);
      }
      float red_buf0[1];
      unsigned int mask[1];
      float t0[1];
      float red_buf0_1[1];
      unsigned int mask_1[1];
      float t0_1[1];
      red_buf0_1[0] = normal_reduce_temp0[0];
      mask_1[0] = __activemask();
      t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
      red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
      t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
      red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
      t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
      red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
      t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
      red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
      t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
      red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
      if ((((int)threadIdx.x) % 32) == 0) {
          union_shared_0_[(((int)threadIdx.x) >> 5)] = red_buf0_1[0];
      }
      asm ("bar.sync 0, 128;");
      if (((int)threadIdx.x) < 4) {
          red_buf0[0] = union_shared_0_[((int)threadIdx.x)];
      }
      mask[0] = (__activemask() & (unsigned int)15);
      t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
      red_buf0[0] = max(red_buf0[0], t0[0]);
      t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
      red_buf0[0] = max(red_buf0[0], t0[0]);
      if (((int)threadIdx.x) == 0) {
          ((volatile float *)union_shared_2_)[0] = red_buf0[0];
      }
      asm ("bar.sync 0, 128;");
      if (((int)threadIdx.x) == 0) {
          union_shared_3_[0] = ((volatile float *)union_shared_2_)[0];
      }
      normal_reduce_temp0_1[0] = 0.F;
      asm ("bar.sync 0, 128;");
      for (int k_outer = 0; k_outer < 8; ++k_outer) {
          if (((k_outer * 16) + (((int)threadIdx.x) >> 3)) < 125) {
              normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_shared_12_data_[(((((int)blockIdx_x_) * 1000) + (k_outer * 128)) + ((int)threadIdx.x))] - union_shared_3_[0])));
          }
      }
      float red_buf0_2[1];
      unsigned int mask_2[1];
      float t0_2[1];
      float red_buf0_3[1];
      unsigned int mask_3[1];
      float t0_3[1];
      red_buf0_3[0] = normal_reduce_temp0_1[0];
      mask_3[0] = __activemask();
      t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
      red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
      t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
      red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
      t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
      red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
      t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
      red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
      t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
      red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
      if ((((int)threadIdx.x) % 32) == 0) {
          union_shared_1_[(((int)threadIdx.x) >> 5)] = red_buf0_3[0];
      }
      asm ("bar.sync 0, 128;");
      if (((int)threadIdx.x) < 4) {
          red_buf0_2[0] = union_shared_1_[((int)threadIdx.x)];
      }
      mask_2[0] = (__activemask() & (unsigned int)15);
      t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
      red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
      t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
      red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
      if (((int)threadIdx.x) == 0) {
          ((volatile float *)union_shared_4_)[0] = red_buf0_2[0];
      }
      asm ("bar.sync 0, 128;");
      if (((int)threadIdx.x) == 0) {
          union_shared_5_[0] = ((volatile float *)union_shared_4_)[0];
      }
      asm ("bar.sync 0, 128;");
      for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
          if (((i2_outer * 16) + (((int)threadIdx.x) >> 3)) < 125) {
              softmax_shared_12_T_softmax_norm_[(((((int)blockIdx_x_) * 1000) + (i2_outer * 128)) + ((int)threadIdx.x))] = (__expf((softmax_shared_12_data_[(((((int)blockIdx_x_) * 1000) + (i2_outer * 128)) + ((int)threadIdx.x))] - union_shared_3_[0])) / union_shared_5_[0]);
          }
      }
  }
  // softmax_shared_12_copy
  else if ((KernelID_ == 1) && ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y >= 0 && threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y < 128)))
  {
      float normal_reduce_temp0[1];
      float normal_reduce_temp0_1[1];
      normal_reduce_temp0[0] = -3.40282306E+38F;
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_copy_data_[((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x))]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_copy_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 128)]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_copy_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 256)]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_copy_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 384)]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_copy_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 512)]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_copy_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 640)]);
      normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_copy_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 768)]);
      if (((int)threadIdx.x) < 104) {
          normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_shared_12_copy_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx.x)) + 896)]);
      }
      float red_buf0[1];
      unsigned int mask[1];
      float t0[1];
      float red_buf0_1[1];
      unsigned int mask_1[1];
      float t0_1[1];
      red_buf0_1[0] = normal_reduce_temp0[0];
      mask_1[0] = __activemask();
      t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 16, 32);
      red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
      t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 8, 32);
      red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
      t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 4, 32);
      red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
      t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 2, 32);
      red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
      t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
      red_buf0_1[0] = max(red_buf0_1[0], t0_1[0]);
      if ((((int)threadIdx.x) % 32) == 0) {
          union_shared_0_[(((int)threadIdx.x) >> 5)] = red_buf0_1[0];
      }
      asm ("bar.sync 0, 128;");
      if (((int)threadIdx.x) < 4) {
          red_buf0[0] = union_shared_0_[((int)threadIdx.x)];
      }
      mask[0] = (__activemask() & (unsigned int)15);
      t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
      red_buf0[0] = max(red_buf0[0], t0[0]);
      t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
      red_buf0[0] = max(red_buf0[0], t0[0]);
      if (((int)threadIdx.x) == 0) {
          ((volatile float *)union_shared_2_)[0] = red_buf0[0];
      }
      asm ("bar.sync 0, 128;");
      if (((int)threadIdx.x) == 0) {
          union_shared_3_[0] = ((volatile float *)union_shared_2_)[0];
      }
      normal_reduce_temp0_1[0] = 0.F;
      asm ("bar.sync 0, 128;");
      for (int k_outer = 0; k_outer < 8; ++k_outer) {
          if (((k_outer * 16) + (((int)threadIdx.x) >> 3)) < 125) {
              normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_shared_12_copy_data_[(((((int)blockIdx_x_) * 1000) + (k_outer * 128)) + ((int)threadIdx.x))] - union_shared_3_[0])));
          }
      }
      float red_buf0_2[1];
      unsigned int mask_2[1];
      float t0_2[1];
      float red_buf0_3[1];
      unsigned int mask_3[1];
      float t0_3[1];
      red_buf0_3[0] = normal_reduce_temp0_1[0];
      mask_3[0] = __activemask();
      t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 16, 32);
      red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
      t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 8, 32);
      red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
      t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 4, 32);
      red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
      t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 2, 32);
      red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
      t0_3[0] = __shfl_down_sync(mask_3[0], red_buf0_3[0], 1, 32);
      red_buf0_3[0] = (red_buf0_3[0] + t0_3[0]);
      if ((((int)threadIdx.x) % 32) == 0) {
          union_shared_1_[(((int)threadIdx.x) >> 5)] = red_buf0_3[0];
      }
      asm ("bar.sync 0, 128;");
      if (((int)threadIdx.x) < 4) {
          red_buf0_2[0] = union_shared_1_[((int)threadIdx.x)];
      }
      mask_2[0] = (__activemask() & (unsigned int)15);
      t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 2, 32);
      red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
      t0_2[0] = __shfl_down_sync(mask_2[0], red_buf0_2[0], 1, 32);
      red_buf0_2[0] = (red_buf0_2[0] + t0_2[0]);
      if (((int)threadIdx.x) == 0) {
          ((volatile float *)union_shared_4_)[0] = red_buf0_2[0];
      }
      asm ("bar.sync 0, 128;");
      if (((int)threadIdx.x) == 0) {
          union_shared_5_[0] = ((volatile float *)union_shared_4_)[0];
      }
      asm ("bar.sync 0, 128;");
      for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
          if (((i2_outer * 16) + (((int)threadIdx.x) >> 3)) < 125) {
              softmax_shared_12_copy_T_softmax_norm_[(((((int)blockIdx_x_) * 1000) + (i2_outer * 128)) + ((int)threadIdx.x))] = (__expf((softmax_shared_12_copy_data_[(((((int)blockIdx_x_) * 1000) + (i2_outer * 128)) + ((int)threadIdx.x))] - union_shared_3_[0])) / union_shared_5_[0]);
          }
      }
  }
}
