
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
extern "C" __global__ __launch_bounds__(112) conv2d_B1_matmul_B16_fused_(float *__restrict conv2d_B1_data_, float *__restrict conv2d_B1_kernel_, float *__restrict conv2d_B1_conv2d_nhwc_, float *__restrict matmul_B16_data_, float *__restrict matmul_B16_weight_, float *__restrict matmul_B16_T_matmul_NT_)
{
  /*
   * KernelID_ means...
   * 0: conv2d_B1
   * 1: matmul_B16
   */
  int gridDim_x_;
  int blockIdx_x_;
  int Others_;
  int KernelID_;
  
  if (blockIdx.x >= 0 && blockIdx.x < 84)
  {
    gridDim_x_ = 224;
    Others_    = 0;
    KernelID_  = 0;
  }
  else if (blockIdx.x >= 84 && blockIdx.x < 164)
  {
    gridDim_x_ = 80;
    Others_    = 84;
    KernelID_  = 1;
  }
  else if (blockIdx.x >= 164 && blockIdx.x < 304)
  {
    gridDim_x_ = 224;
    Others_    = 80;
    KernelID_  = 0;
  }
  blockIdx_x_ = blockIdx.x - Others_;
  
  // conv2d_B1
  if ((KernelID_ == 0) && ((threadIdx.x >= 0 && threadIdx.x < 112)))
  {
      float conv2d_nhwc_local[4];
      conv2d_nhwc_local[0] = conv2d_B1_data_[0];
      int a = blockIdx_x_;
      int b = gridDim_x_;
      asm ("bar.sync 0, 112;");
  }
  // matmul_B16
  else if ((KernelID_ == 1) && ((threadIdx.x >= 0 && threadIdx.x < 50)))
  {
      float T_matmul_NT_local[4];
      T_matmul_NT_local[0] = matmul_B16_data_[0];
      int a = blockIdx_x_;
      int b = gridDim_x_;
      asm ("bar.sync 0, 50;");
  }
}
