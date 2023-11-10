
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
template <typename conv2d_B1_test_TypeA_, typename conv2d_B1_test_TypeB_, typename matmul_B16_test_TypeA_, typename matmul_B16_test_TypeB_>
extern "C" __global__ __launch_bounds__(112) void conv2d_B1_test_matmul_B16_test_fused_kernel_bfuse_idx_0(float *__restrict conv2d_B1_test_data_, float *__restrict conv2d_B1_test_kernel_, float *__restrict conv2d_B1_test_conv2d_nhwc_, float *__restrict matmul_B16_test_data_, float *__restrict matmul_B16_test_weight_, float *__restrict matmul_B16_test_T_matmul_NT_)
{
  /*
   * KernelID_ means...
   * 0: conv2d_B1_test
   * 1: matmul_B16_test
   */
  int gridDim_x_, gridDim_y_, gridDim_z_;
  int blockIdx_x_, blockIdx_y_, blockIdx_z_;
  int TotalBlockIdx_;
  int KernelID_;
  
  if (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y >= 0 && blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y < 224)
  {
    TotalBlockIdx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y - 0;
    KernelID_  = 0;
    gridDim_x_ = 224;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
  }
  else if (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y >= 224 && blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y < 304)
  {
    TotalBlockIdx_ = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y - 224;
    KernelID_  = 1;
    gridDim_x_ = 80;
    gridDim_y_ = 1;
    gridDim_z_ = 1;
  }
  blockIdx_x_ = TotalBlockIdx_ % gridDim_x_;
  blockIdx_y_ = TotalBlockIdx_ / gridDim_x_ % gridDim_y_;
  blockIdx_z_ = TotalBlockIdx_ / (gridDim_x_ * gridDim_y_);

  static float union_shared_0_[3200] __attribute__((shared));
  static float union_shared_1_[2304] __attribute__((shared));


  // conv2d_B1_test
  if ((KernelID_ == 0) && ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y >= 0 && threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y < 112)))
  {
      float conv2d_nhwc_local[4];
      conv2d_nhwc_local[0] = conv2d_B1_test_data_[0];
      conv2d_nhwc_local[0] = union_shared_0_[0];
      int a = blockIdx_x_;
      int b = gridDim_x_;
      asm ("bar.sync 0, 112;");
      conv2d_B1_test_TypeA_ A;
      conv2d_B1_test_TypeB_ B;
      A = B;
  }
  // matmul_B16_test
  else if ((KernelID_ == 1) && ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y >= 0 && threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y < 50)))
  {
      float T_matmul_NT_local[4];
      T_matmul_NT_local[0] = matmul_B16_test_data_[0];
      T_matmul_NT_local[0] = union_shared_1_[0];
      int a = blockIdx_x_;
      int b = gridDim_x_;
      asm ("bar.sync 0, 50;");
      matmul_B16_test_TypeA_ A;
      matmul_B16_test_TypeB_ B;
      A = B;
  }
}
