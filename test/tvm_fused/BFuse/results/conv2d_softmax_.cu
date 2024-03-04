

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

extern "C" __global__ __launch_bounds__(232) void conv2d_softmax_fused_kernel_bfuse_idx_0(float *__restrict conv2d_conv2d_nchw_, float *__restrict conv2d_data_, float *__restrict conv2d_kernel_, float *__restrict softmax_T_softmax_norm_, float *__restrict softmax_data_)
{
  /*
   * KernelID_ means...
   * 0: conv2d
   * 1: softmax
   */
  int gridDim_x_, gridDim_y_, gridDim_z_;
  int blockIdx_x_, blockIdx_y_, blockIdx_z_;
  int blockDim_x_, blockDim_y_, blockDim_z_;
  int threadIdx_x_, threadIdx_y_, threadIdx_z_;
  int NewBlockIdx_;
  int KernelID_;
  
    else   else if (((int)blockIdx.x >= 0 && (int)blockIdx.x < 504) && ((((int)blockIdx.x - 0) / 84) % 1 == 0))
  {
    NewBlockIdx_ = 0 + (((int)blockIdx.x - 0) / 84) * 84 + (int)blockIdx.x % 84;
    KernelID_  = 1;
    gridDim_x_ = 512;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
    blockDim_x_ = 50;
    blockDim_y_ = 1;
    blockDim_z_ = 1;
  }
  else if ((int)blockIdx.x >= 504 && (int)blockIdx.x < 562)
  {
    NewBlockIdx_ = (int)blockIdx.x - 504;
    KernelID_  = 0;
    gridDim_x_ = 58;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
    blockDim_x_ = 232;
    blockDim_y_ = 1;
    blockDim_z_ = 1;
  }
  else if ((int)blockIdx.x >= 562 && (int)blockIdx.x < 570)
  {
    NewBlockIdx_ = (int)blockIdx.x - 58;
    KernelID_  = 1;
    gridDim_x_ = 512;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
    blockDim_x_ = 50;
    blockDim_y_ = 1;
    blockDim_z_ = 1;
  }
  blockIdx_x_ = NewBlockIdx_ % gridDim_x_;
  blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;
  blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);
  threadIdx_x_ = (int)threadIdx.x % blockDim_x_;
  threadIdx_y_ = (int)threadIdx.x / blockDim_x_ % blockDim_y_;
  threadIdx_z_ = (int)threadIdx.x / (blockDim_x_ * blockDim_y_);

  static float union_shared_0_[7424] __attribute__((shared));
  static float union_shared_1_[2048] __attribute__((shared));
  static float union_shared_2_[1] __attribute__((shared));
  static float union_shared_3_[1] __attribute__((shared));


  // conv2d
  if ((KernelID_ == 0) && ((int)threadIdx.x < 232))
  {
      float conv2d_nchw_local[16];
      for (int ff_c_inner_init = 0; ff_c_inner_init < 2; ++ff_c_inner_init) {
          conv2d_nchw_local[ff_c_inner_init] = 0.F;
          conv2d_nchw_local[(ff_c_inner_init + 2)] = 0.F;
          conv2d_nchw_local[(ff_c_inner_init + 4)] = 0.F;
          conv2d_nchw_local[(ff_c_inner_init + 6)] = 0.F;
          conv2d_nchw_local[(ff_c_inner_init + 8)] = 0.F;
          conv2d_nchw_local[(ff_c_inner_init + 10)] = 0.F;
          conv2d_nchw_local[(ff_c_inner_init + 12)] = 0.F;
          conv2d_nchw_local[(ff_c_inner_init + 14)] = 0.F;
      }
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 32; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
          union_shared_0_[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 232) + ((int)threadIdx_x_))] = (((((1 <= (((((int)blockIdx_x_) % 29) * 2) + ((((int)threadIdx_x_) % 116) / 58))) && ((((((int)blockIdx_x_) % 29) * 2) + ((((int)threadIdx_x_) % 116) / 58)) < 57)) && (1 <= (((int)threadIdx_x_) % 58))) && ((((int)threadIdx_x_) % 58) < 57)) ? conv2d_data_[((((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 6272) + ((((int)threadIdx_x_) / 116) * 3136)) + ((((int)blockIdx_x_) % 29) * 112)) + (((((int)threadIdx_x_) % 116) / 58) * 56)) + (((int)threadIdx_x_) % 58)) - 57)] : 0.F);
      }
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1) {
          if (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 29) + (((int)threadIdx_x_) >> 3)) < 128) {
              *(float2 *)(union_shared_1_ + ((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 464) + (((int)threadIdx_x_) * 2))) = *(float2 *)(conv2d_kernel_ + ((((((int)blockIdx_x_) / 29) * 2048) + (ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer_1 * 464)) + (((int)threadIdx_x_) * 2)));
          }
      }
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
          for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
              for (int ff_c_inner = 0; ff_c_inner < 2; ++ff_c_inner) {
                  conv2d_nchw_local[ff_c_inner] = (conv2d_nchw_local[ff_c_inner] + (union_shared_0_[(((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx_x_) % 58))] * union_shared_1_[(((((((int)threadIdx_x_) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner)]));
                  conv2d_nchw_local[(ff_c_inner + 2)] = (conv2d_nchw_local[(ff_c_inner + 2)] + (union_shared_0_[((((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx_x_) % 58)) + 58)] * union_shared_1_[(((((((int)threadIdx_x_) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner)]));
                  conv2d_nchw_local[(ff_c_inner + 4)] = (conv2d_nchw_local[(ff_c_inner + 4)] + (union_shared_0_[(((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx_x_) % 58))] * union_shared_1_[((((((((int)threadIdx_x_) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 512)]));
                  conv2d_nchw_local[(ff_c_inner + 6)] = (conv2d_nchw_local[(ff_c_inner + 6)] + (union_shared_0_[((((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx_x_) % 58)) + 58)] * union_shared_1_[((((((((int)threadIdx_x_) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 512)]));
                  conv2d_nchw_local[(ff_c_inner + 8)] = (conv2d_nchw_local[(ff_c_inner + 8)] + (union_shared_0_[(((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx_x_) % 58))] * union_shared_1_[((((((((int)threadIdx_x_) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 1024)]));
                  conv2d_nchw_local[(ff_c_inner + 10)] = (conv2d_nchw_local[(ff_c_inner + 10)] + (union_shared_0_[((((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx_x_) % 58)) + 58)] * union_shared_1_[((((((((int)threadIdx_x_) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 1024)]));
                  conv2d_nchw_local[(ff_c_inner + 12)] = (conv2d_nchw_local[(ff_c_inner + 12)] + (union_shared_0_[(((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx_x_) % 58))] * union_shared_1_[((((((((int)threadIdx_x_) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 1536)]));
                  conv2d_nchw_local[(ff_c_inner + 14)] = (conv2d_nchw_local[(ff_c_inner + 14)] + (union_shared_0_[((((rc_outer_inner * 1856) + (rc_inner * 116)) + (((int)threadIdx_x_) % 58)) + 58)] * union_shared_1_[((((((((int)threadIdx_x_) / 58) * 128) + (ff_c_inner * 64)) + (rc_outer_inner * 16)) + rc_inner) + 1536)]));
              }
          }
      }
      for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
          conv2d_conv2d_nchw_[((((((((int)blockIdx_x_) / 29) * 107648) + ((((int)threadIdx_x_) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx_x_) % 29) * 116)) + (((int)threadIdx_x_) % 58))] = conv2d_nchw_local[ff_inner];
          conv2d_conv2d_nchw_[(((((((((int)blockIdx_x_) / 29) * 107648) + ((((int)threadIdx_x_) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx_x_) % 29) * 116)) + (((int)threadIdx_x_) % 58)) + 58)] = conv2d_nchw_local[(ff_inner + 2)];
          conv2d_conv2d_nchw_[(((((((((int)blockIdx_x_) / 29) * 107648) + ((((int)threadIdx_x_) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx_x_) % 29) * 116)) + (((int)threadIdx_x_) % 58)) + 26912)] = conv2d_nchw_local[(ff_inner + 4)];
          conv2d_conv2d_nchw_[(((((((((int)blockIdx_x_) / 29) * 107648) + ((((int)threadIdx_x_) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx_x_) % 29) * 116)) + (((int)threadIdx_x_) % 58)) + 26970)] = conv2d_nchw_local[(ff_inner + 6)];
          conv2d_conv2d_nchw_[(((((((((int)blockIdx_x_) / 29) * 107648) + ((((int)threadIdx_x_) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx_x_) % 29) * 116)) + (((int)threadIdx_x_) % 58)) + 53824)] = conv2d_nchw_local[(ff_inner + 8)];
          conv2d_conv2d_nchw_[(((((((((int)blockIdx_x_) / 29) * 107648) + ((((int)threadIdx_x_) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx_x_) % 29) * 116)) + (((int)threadIdx_x_) % 58)) + 53882)] = conv2d_nchw_local[(ff_inner + 10)];
          conv2d_conv2d_nchw_[(((((((((int)blockIdx_x_) / 29) * 107648) + ((((int)threadIdx_x_) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx_x_) % 29) * 116)) + (((int)threadIdx_x_) % 58)) + 80736)] = conv2d_nchw_local[(ff_inner + 12)];
          conv2d_conv2d_nchw_[(((((((((int)blockIdx_x_) / 29) * 107648) + ((((int)threadIdx_x_) / 58) * 6728)) + (ff_inner * 3364)) + ((((int)blockIdx_x_) % 29) * 116)) + (((int)threadIdx_x_) % 58)) + 80794)] = conv2d_nchw_local[(ff_inner + 14)];
      }
  }
  // softmax
  else if ((KernelID_ == 1) && ((int)threadIdx.x < 50))
  {
      float normal_reduce_temp0[1];
      float normal_reduce_temp0_1[1];
      normal_reduce_temp0[0] = -3.40282306E+38F;
      for (int k_outer = 0; k_outer < 20; ++k_outer) {
          normal_reduce_temp0[0] = max(normal_reduce_temp0[0], softmax_data_[(((((int)blockIdx_x_) * 1000) + (k_outer * 50)) + ((int)threadIdx_x_))]);
      }
      __syncthreads();
      ((volatile float *)union_shared_0_)[((int)threadIdx_x_)] = normal_reduce_temp0[0];
      __syncthreads();
      if (((int)threadIdx_x_) < 18) {
          ((volatile float *)union_shared_0_)[((int)threadIdx_x_)] = max(((volatile float *)union_shared_0_)[((int)threadIdx_x_)], ((volatile float *)union_shared_0_)[(((int)threadIdx_x_) + 32)]);
      }
      __syncthreads();
      if (((int)threadIdx_x_) < 16) {
          float w_16_0 = max(((volatile float *)union_shared_0_)[((int)threadIdx_x_)], ((volatile float *)union_shared_0_)[(((int)threadIdx_x_) + 16)]);
          ((volatile float *)union_shared_0_)[((int)threadIdx_x_)] = w_16_0;
          float w_8_0 = max(((volatile float *)union_shared_0_)[((int)threadIdx_x_)], ((volatile float *)union_shared_0_)[(((int)threadIdx_x_) + 8)]);
          ((volatile float *)union_shared_0_)[((int)threadIdx_x_)] = w_8_0;
          float w_4_0 = max(((volatile float *)union_shared_0_)[((int)threadIdx_x_)], ((volatile float *)union_shared_0_)[(((int)threadIdx_x_) + 4)]);
          ((volatile float *)union_shared_0_)[((int)threadIdx_x_)] = w_4_0;
          float w_2_0 = max(((volatile float *)union_shared_0_)[((int)threadIdx_x_)], ((volatile float *)union_shared_0_)[(((int)threadIdx_x_) + 2)]);
          ((volatile float *)union_shared_0_)[((int)threadIdx_x_)] = w_2_0;
          float w_1_0 = max(((volatile float *)union_shared_0_)[((int)threadIdx_x_)], ((volatile float *)union_shared_0_)[(((int)threadIdx_x_) + 1)]);
          ((volatile float *)union_shared_0_)[((int)threadIdx_x_)] = w_1_0;
      }
      __syncthreads();
      if (((int)threadIdx_x_) == 0) {
          union_shared_2_[0] = ((volatile float *)union_shared_0_)[0];
      }
      normal_reduce_temp0_1[0] = 0.F;
      __syncthreads();
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_))] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 50)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 100)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 150)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 200)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 250)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 300)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 350)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 400)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 450)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 500)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 550)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 600)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 650)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 700)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 750)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 800)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 850)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 900)] - union_shared_2_[0])));
      normal_reduce_temp0_1[0] = (normal_reduce_temp0_1[0] + __expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + ((int)threadIdx_x_)) + 950)] - union_shared_2_[0])));
      __syncthreads();
      ((volatile float *)union_shared_1_)[((int)threadIdx_x_)] = normal_reduce_temp0_1[0];
      __syncthreads();
      if (((int)threadIdx_x_) < 18) {
          ((volatile float *)union_shared_1_)[((int)threadIdx_x_)] = (((volatile float *)union_shared_1_)[((int)threadIdx_x_)] + ((volatile float *)union_shared_1_)[(((int)threadIdx_x_) + 32)]);
      }
      __syncthreads();
      if (((int)threadIdx_x_) < 16) {
          float w_16_0_1 = (((volatile float *)union_shared_1_)[((int)threadIdx_x_)] + ((volatile float *)union_shared_1_)[(((int)threadIdx_x_) + 16)]);
          ((volatile float *)union_shared_1_)[((int)threadIdx_x_)] = w_16_0_1;
          float w_8_0_1 = (((volatile float *)union_shared_1_)[((int)threadIdx_x_)] + ((volatile float *)union_shared_1_)[(((int)threadIdx_x_) + 8)]);
          ((volatile float *)union_shared_1_)[((int)threadIdx_x_)] = w_8_0_1;
          float w_4_0_1 = (((volatile float *)union_shared_1_)[((int)threadIdx_x_)] + ((volatile float *)union_shared_1_)[(((int)threadIdx_x_) + 4)]);
          ((volatile float *)union_shared_1_)[((int)threadIdx_x_)] = w_4_0_1;
          float w_2_0_1 = (((volatile float *)union_shared_1_)[((int)threadIdx_x_)] + ((volatile float *)union_shared_1_)[(((int)threadIdx_x_) + 2)]);
          ((volatile float *)union_shared_1_)[((int)threadIdx_x_)] = w_2_0_1;
          float w_1_0_1 = (((volatile float *)union_shared_1_)[((int)threadIdx_x_)] + ((volatile float *)union_shared_1_)[(((int)threadIdx_x_) + 1)]);
          ((volatile float *)union_shared_1_)[((int)threadIdx_x_)] = w_1_0_1;
      }
      __syncthreads();
      if (((int)threadIdx_x_) == 0) {
          union_shared_3_[0] = ((volatile float *)union_shared_1_)[0];
      }
      __syncthreads();
      for (int i1_outer = 0; i1_outer < 20; ++i1_outer) {
          softmax_T_softmax_norm_[(((((int)blockIdx_x_) * 1000) + (i1_outer * 50)) + ((int)threadIdx_x_))] = (__expf((softmax_data_[(((((int)blockIdx_x_) * 1000) + (i1_outer * 50)) + ((int)threadIdx_x_))] - union_shared_2_[0])) / union_shared_3_[0]);
      }
  }
}
