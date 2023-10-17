template <typename conv2d_B1_TypeA_, typename conv2d_B1_TypeB_, typename matmul_B16_TypeA_, typename matmul_B16_TypeB_>
__global__ __launch_bounds__(112) void conv2d_B1_matmul_B16_fused_kernel_hfuse_idx_0(float *__restrict conv2d_B1_data_, float *__restrict conv2d_B1_kernel_, float *__restrict conv2d_B1_conv2d_nhwc_, float *__restrict matmul_B16_data_, float *__restrict matmul_B16_weight_, float *__restrict matmul_B16_T_matmul_NT_)
{
  static float union_shared_0_[3200] __attribute__((shared));
  static float union_shared_1_[2304] __attribute__((shared));

  /*
   * KernelID_ means...
   * 0: conv2d_B1
   * 1: matmul_B16
   */
  int gridDim_x_, gridDim_y_, gridDim_z_;
  int blockIdx_x_, blockIdx_y_, blockIdx_z_;
  int TotalBlockIdx_;
  int KernelID_;
  
  if (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y >= 0 && blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y < 84)
  {
    TotalBlockIdx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y - 0;
    KernelID_      = 0;
    gridDim_x_ = 224;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
  }
  else if (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y >= 84 && blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y < 164)
  {
    TotalBlockIdx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y - 84;
    KernelID_      = 1;
    gridDim_x_ = 80;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
  }
  else if (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y >= 164 && blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y < 304)
  {
    TotalBlockIdx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y - 80;
    KernelID_      = 0;
    gridDim_x_ = 224;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
  }
  blockIdx_x_ = TotalBlockIdx_ % gridDim_x_;
  blockIdx_y_ = TotalBlockIdx_ / gridDim_x_ % gridDim_y_;
  blockIdx_z_ = TotalBlockIdx_ / (gridDim_x_ * gridDim_y_);
  
  // conv2d_B1
  if ((KernelID_ == 0) && ((threadIdx.x + threadIdx.y * threadIdx.x + threadIdx.z * threadIdx.y * threadIdx.z >= 0 && threadIdx.x + threadIdx.y * threadIdx.x + threadIdx.z * threadIdx.y * threadIdx.z < 112)))
  {
      float conv2d_nhwc_local[4];
      conv2d_nhwc_local[0] = conv2d_B1_data_[0];
      conv2d_nhwc_local[0] = union_shared_0_[0];
      int a = blockIdx_x_;
      int b = gridDim_x_;
      asm ("bar.sync 0, 112;");
      conv2d_B1_TypeA_ A;
      conv2d_B1_TypeB_ B;
      A = B;
  }
  // matmul_B16
  else if ((KernelID_ == 1) && ((threadIdx.x + threadIdx.y * threadIdx.x + threadIdx.z * threadIdx.y * threadIdx.z >= 0 && threadIdx.x + threadIdx.y * threadIdx.x + threadIdx.z * threadIdx.y * threadIdx.z < 50)))
  {
      float T_matmul_NT_local[4];
      T_matmul_NT_local[0] = matmul_B16_data_[0];
      T_matmul_NT_local[0] = union_shared_1_[0];
      int a = blockIdx_x_;
      int b = gridDim_x_;
      asm ("bar.sync 0, 50;");
      matmul_B16_TypeA_ A;
      matmul_B16_TypeB_ B;
      A = B;
  }
}
