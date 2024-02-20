

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

extern "C" __global__ __launch_bounds__(128) void bgemm_shared_8192_bgemm_shared_8192_copy_fused_kernel_bfuse_idx_0(float *__restrict bgemm_shared_8192_A_, float *__restrict bgemm_shared_8192_B_, float *__restrict bgemm_shared_8192_T_batch_matmul_NT_, float *__restrict bgemm_shared_8192_copy_A_, float *__restrict bgemm_shared_8192_copy_B_, float *__restrict bgemm_shared_8192_copy_T_batch_matmul_NT_)
{
  /*
   * KernelID_ means...
   * 0: bgemm_shared_8192
   * 1: bgemm_shared_8192_copy
   */
  int gridDim_x_;
  int blockIdx_x_;
  int NewBlockIdx_;
  int KernelID_;
  
  if (((int)blockIdx.x >= 0 && (int)blockIdx.x < 65520) && ((((int)blockIdx.x - 0) / 84) % 2 == 0))
  {
    NewBlockIdx_ = 0 + ((int)blockIdx.x - (int)blockIdx.x % 168) / 2 + (int)blockIdx.x % 84;
    KernelID_  = 0;
    gridDim_x_ = 32768;
  }
  else if (((int)blockIdx.x >= 0 && (int)blockIdx.x < 65520) && ((((int)blockIdx.x - 0) / 84) % 2 == 1))
  {
    NewBlockIdx_ = 0 + ((int)blockIdx.x - (int)blockIdx.x % 168) / 2 + (int)blockIdx.x % 84;
    KernelID_  = 1;
    gridDim_x_ = 32768;
  }
  else if ((int)blockIdx.x >= 65520 && (int)blockIdx.x < 65528)
  {
    NewBlockIdx_ = (int)blockIdx.x - 32760;
    KernelID_  = 0;
    gridDim_x_ = 32768;
  }
  else if ((int)blockIdx.x >= 65528 && (int)blockIdx.x < 65536)
  {
    NewBlockIdx_ = (int)blockIdx.x - 32768;
    KernelID_  = 1;
    gridDim_x_ = 32768;
  }
  blockIdx_x_ = NewBlockIdx_;

  static float union_shared_0_[4096] __attribute__((shared));
  static float union_shared_1_[4096] __attribute__((shared));


  // bgemm_shared_8192
  if ((KernelID_ == 0) && (((int)threadIdx.x >= 0 && (int)threadIdx.x < 128)))
  {
      float T_batch_matmul_NT_local[32];
      for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
          T_batch_matmul_NT_local[(i_c_outer_inner_init * 8)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 16)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 1)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 17)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 2)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 18)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 3)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 19)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 4)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 20)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 5)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 21)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 6)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 22)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 7)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 23)] = 0.F;
      }
      for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
          asm ("bar.sync 0, 128;");
          *(float2 *)(union_shared_0_ + (((int)threadIdx.x) * 2)) = *(float2 *)(bgemm_shared_8192_A_ + (((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 256)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 8192));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 512)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 16384));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 768)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 24576));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 1024)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 32768));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 1280)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 40960));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 1536)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 49152));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 1792)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 57344));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 2048)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 65536));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 2304)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 73728));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 2560)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 81920));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 2816)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 90112));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 3072)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 98304));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 3328)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 106496));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 3584)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 114688));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 3840)) = *(float2 *)(bgemm_shared_8192_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 122880));
          union_shared_1_[((int)threadIdx.x)] = bgemm_shared_8192_B_[((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
          union_shared_1_[(((int)threadIdx.x) + 128)] = bgemm_shared_8192_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
          union_shared_1_[(((int)threadIdx.x) + 256)] = bgemm_shared_8192_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192)];
          union_shared_1_[(((int)threadIdx.x) + 384)] = bgemm_shared_8192_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288)];
          union_shared_1_[(((int)threadIdx.x) + 512)] = bgemm_shared_8192_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 16384)];
          union_shared_1_[(((int)threadIdx.x) + 640)] = bgemm_shared_8192_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 20480)];
          union_shared_1_[(((int)threadIdx.x) + 768)] = bgemm_shared_8192_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 24576)];
          union_shared_1_[(((int)threadIdx.x) + 896)] = bgemm_shared_8192_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 28672)];
          asm ("bar.sync 0, 128;");
          for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
              for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
                  T_batch_matmul_NT_local[(i_c_outer_inner * 8)] = (T_batch_matmul_NT_local[(i_c_outer_inner * 8)] + (union_shared_0_[((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 16)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 16)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 1)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 1)] + (union_shared_0_[((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 17)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 17)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 2)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 2)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 18)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 18)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 3)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 3)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 19)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 19)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 4)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 4)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 20)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 20)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 5)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 5)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 21)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 21)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 6)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 6)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 22)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 22)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 7)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 7)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 23)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 23)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
              }
          }
      }
      for (int i_inner = 0; i_inner < 8; ++i_inner) {
          for (int j_inner = 0; j_inner < 2; ++j_inner) {
              bgemm_shared_8192_T_batch_matmul_NT_[(((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx_x_) & 31) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + j_inner)] = T_batch_matmul_NT_local[((i_inner * 2) + j_inner)];
              bgemm_shared_8192_T_batch_matmul_NT_[((((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx_x_) & 31) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + j_inner) + 65536)] = T_batch_matmul_NT_local[(((i_inner * 2) + j_inner) + 16)];
          }
      }
  }
  // bgemm_shared_8192_copy
  else if ((KernelID_ == 1) && (((int)threadIdx.x >= 0 && (int)threadIdx.x < 128)))
  {
      float T_batch_matmul_NT_local[32];
      for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
          T_batch_matmul_NT_local[(i_c_outer_inner_init * 8)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 16)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 1)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 17)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 2)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 18)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 3)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 19)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 4)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 20)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 5)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 21)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 6)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 22)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 7)] = 0.F;
          T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 23)] = 0.F;
      }
      for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
          asm ("bar.sync 0, 128;");
          *(float2 *)(union_shared_0_ + (((int)threadIdx.x) * 2)) = *(float2 *)(bgemm_shared_8192_copy_A_ + (((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 256)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 8192));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 512)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 16384));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 768)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 24576));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 1024)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 32768));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 1280)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 40960));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 1536)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 49152));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 1792)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 57344));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 2048)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 65536));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 2304)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 73728));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 2560)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 81920));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 2816)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 90112));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 3072)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 98304));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 3328)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 106496));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 3584)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 114688));
          *(float2 *)(union_shared_0_ + ((((int)threadIdx.x) * 2) + 3840)) = *(float2 *)(bgemm_shared_8192_copy_A_ + ((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 122880));
          union_shared_1_[((int)threadIdx.x)] = bgemm_shared_8192_copy_B_[((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
          union_shared_1_[(((int)threadIdx.x) + 128)] = bgemm_shared_8192_copy_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
          union_shared_1_[(((int)threadIdx.x) + 256)] = bgemm_shared_8192_copy_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192)];
          union_shared_1_[(((int)threadIdx.x) + 384)] = bgemm_shared_8192_copy_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288)];
          union_shared_1_[(((int)threadIdx.x) + 512)] = bgemm_shared_8192_copy_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 16384)];
          union_shared_1_[(((int)threadIdx.x) + 640)] = bgemm_shared_8192_copy_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 20480)];
          union_shared_1_[(((int)threadIdx.x) + 768)] = bgemm_shared_8192_copy_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 24576)];
          union_shared_1_[(((int)threadIdx.x) + 896)] = bgemm_shared_8192_copy_B_[(((((((((int)blockIdx_x_) >> 8) * 1048576) + ((((int)blockIdx_x_) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 28672)];
          asm ("bar.sync 0, 128;");
          for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
              for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
                  T_batch_matmul_NT_local[(i_c_outer_inner * 8)] = (T_batch_matmul_NT_local[(i_c_outer_inner * 8)] + (union_shared_0_[((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 16)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 16)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 1)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 1)] + (union_shared_0_[((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 17)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 17)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 2)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 2)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 18)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 18)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 3)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 3)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 19)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 19)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 4)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 4)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 20)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 20)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 5)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 5)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 21)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 21)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 6)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 6)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 22)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 22)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * union_shared_1_[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 7)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 7)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
                  T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 23)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 23)] + (union_shared_0_[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * union_shared_1_[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
              }
          }
      }
      for (int i_inner = 0; i_inner < 8; ++i_inner) {
          for (int j_inner = 0; j_inner < 2; ++j_inner) {
              bgemm_shared_8192_copy_T_batch_matmul_NT_[(((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx_x_) & 31) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + j_inner)] = T_batch_matmul_NT_local[((i_inner * 2) + j_inner)];
              bgemm_shared_8192_copy_T_batch_matmul_NT_[((((((((((int)blockIdx_x_) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx_x_) & 31) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + j_inner) + 65536)] = T_batch_matmul_NT_local[(((i_inner * 2) + j_inner) + 16)];
          }
      }
  }
}
