extern "C" __global__ __launch_bounds__(104) void bgemm_0_conv2d_3_fused_bfuse(float *__restrict bgemm_0_A_, float *__restrict bgemm_0_B_, float *__restrict bgemm_0_T_batch_matmul_NT_, float *__restrict conv2d_3_conv2d_nchw_, float *__restrict conv2d_3_data_, float *__restrict conv2d_3_kernel_)
{
  /*
   * KernelID_ means...
   * 0: bgemm_0
   * 1: conv2d_3
   */
  int gridDim_x_, gridDim_y_, gridDim_z_;
  int blockIdx_x_, blockIdx_y_, blockIdx_z_;
  int blockDim_x_, blockDim_y_, blockDim_z_;
  int threadIdx_x_, threadIdx_y_, threadIdx_z_;
  int NewBlockIdx_;
  int KernelID_;
  
  if (((int)blockIdx.x < 131072) && ((int)blockIdx.x % 16 >= 0) && ((int)blockIdx.x % 16 < 8))
  {
    NewBlockIdx_ = int((int)blockIdx.x / 16) * 8 + ((int)blockIdx.x % 16 - 0);
    KernelID_  = 0;
    gridDim_x_ = 8;
    gridDim_y_ = 8;
    gridDim_z_ = 1024;
    blockDim_x_ = 8;
    blockDim_y_ = 8;
    blockDim_z_ = 1;
  }
  else if (((int)blockIdx.x < 131072) && ((int)blockIdx.x % 16 >= 8) && ((int)blockIdx.x % 16 < 16))
  {
    NewBlockIdx_ = int((int)blockIdx.x / 16) * 8 + ((int)blockIdx.x % 16 - 8);
    KernelID_  = 1;
    gridDim_x_ = 2;
    gridDim_y_ = 13;
    gridDim_z_ = 4096;
    blockDim_x_ = 13;
    blockDim_y_ = 1;
    blockDim_z_ = 8;
  }
  else if ((int)blockIdx.x >= 131072 && (int)blockIdx.x < 131072)
  {
    NewBlockIdx_ = (int)blockIdx.x - 131072 + 65536;
    KernelID_  = 0;
    gridDim_x_ = 8;
    gridDim_y_ = 8;
    gridDim_z_ = 1024;
    blockDim_x_ = 8;
    blockDim_y_ = 8;
    blockDim_z_ = 1;
  }
  else if ((int)blockIdx.x >= 131072 && (int)blockIdx.x < 172032)
  {
    NewBlockIdx_ = (int)blockIdx.x - 131072 + 65536;
    KernelID_  = 1;
    gridDim_x_ = 2;
    gridDim_y_ = 13;
    gridDim_z_ = 4096;
    blockDim_x_ = 13;
    blockDim_y_ = 1;
    blockDim_z_ = 8;
  }
  blockIdx_x_ = NewBlockIdx_ % gridDim_x_;
  blockIdx_y_ = NewBlockIdx_ / gridDim_x_ % gridDim_y_;
  blockIdx_z_ = NewBlockIdx_ / (gridDim_x_ * gridDim_y_);
  threadIdx_x_ = (int)threadIdx.x % blockDim_x_;
  threadIdx_y_ = (int)threadIdx.x / blockDim_x_ % blockDim_y_;
  threadIdx_z_ = (int)threadIdx.x / (blockDim_x_ * blockDim_y_);

  typedef struct bgemm_0 {
     float A_shared[512];
     float B_shared[512];
  } bgemm_0Ty_;
  typedef struct conv2d_3 {
     float pad_temp_shared[208];
     float kernel_shared[512];
  } conv2d_3Ty_;
  typedef union ShrdUnion {
    bgemm_0Ty_ bgemm_0;
    conv2d_3Ty_ conv2d_3;
  } ShrdUnionTy_;

  __shared__ ShrdUnionTy_ SU_;

  // bgemm_0
  if ((KernelID_ == 0) && ((int)threadIdx.x < 64))
  {
      float T_batch_matmul_NT_local[64];
      float A_shared_local[8];
      float B_shared_local[8];
      for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
          for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
              T_batch_matmul_NT_local[((i_c_init * 8) + j_c_init)] = 0.F;
          }
      }
      for (int k_outer = 0; k_outer < 8; ++k_outer) {
          __syncthreads();
          SU_.bgemm_0.A_shared[((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_))] = bgemm_0_A_[(((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_))];
          SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 8)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 64)];
          SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 16)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 128)];
          SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 24)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 192)];
          SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 32)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 256)];
          SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 40)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 320)];
          SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 48)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 384)];
          SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 56)] = bgemm_0_A_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_y_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 448)];
          SU_.bgemm_0.B_shared[((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_))] = bgemm_0_B_[(((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_))];
          SU_.bgemm_0.B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 8)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 64)];
          SU_.bgemm_0.B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 16)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 128)];
          SU_.bgemm_0.B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 24)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 192)];
          SU_.bgemm_0.B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 32)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 256)];
          SU_.bgemm_0.B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 40)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 320)];
          SU_.bgemm_0.B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 48)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 384)];
          SU_.bgemm_0.B_shared[(((((int)threadIdx_y_) * 64) + ((int)threadIdx_x_)) + 56)] = bgemm_0_B_[((((((((int)blockIdx_z_) * 32768) + (((int)blockIdx_x_) * 4096)) + (((int)threadIdx_y_) * 512)) + (k_outer * 8)) + ((int)threadIdx_x_)) + 448)];
          __syncthreads();
          for (int k_inner = 0; k_inner < 8; ++k_inner) {
              A_shared_local[0] = SU_.bgemm_0.A_shared[((((int)threadIdx_y_) * 64) + k_inner)];
              A_shared_local[1] = SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 8)];
              A_shared_local[2] = SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 16)];
              A_shared_local[3] = SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 24)];
              A_shared_local[4] = SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 32)];
              A_shared_local[5] = SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 40)];
              A_shared_local[6] = SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 48)];
              A_shared_local[7] = SU_.bgemm_0.A_shared[(((((int)threadIdx_y_) * 64) + k_inner) + 56)];
              B_shared_local[0] = SU_.bgemm_0.B_shared[((((int)threadIdx_x_) * 64) + k_inner)];
              B_shared_local[1] = SU_.bgemm_0.B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 8)];
              B_shared_local[2] = SU_.bgemm_0.B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 16)];
              B_shared_local[3] = SU_.bgemm_0.B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 24)];
              B_shared_local[4] = SU_.bgemm_0.B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 32)];
              B_shared_local[5] = SU_.bgemm_0.B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 40)];
              B_shared_local[6] = SU_.bgemm_0.B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 48)];
              B_shared_local[7] = SU_.bgemm_0.B_shared[(((((int)threadIdx_x_) * 64) + k_inner) + 56)];
              for (int i_c = 0; i_c < 8; ++i_c) {
                  T_batch_matmul_NT_local[(i_c * 8)] = (T_batch_matmul_NT_local[(i_c * 8)] + (A_shared_local[i_c] * B_shared_local[0]));
                  T_batch_matmul_NT_local[((i_c * 8) + 1)] = (T_batch_matmul_NT_local[((i_c * 8) + 1)] + (A_shared_local[i_c] * B_shared_local[1]));
                  T_batch_matmul_NT_local[((i_c * 8) + 2)] = (T_batch_matmul_NT_local[((i_c * 8) + 2)] + (A_shared_local[i_c] * B_shared_local[2]));
                  T_batch_matmul_NT_local[((i_c * 8) + 3)] = (T_batch_matmul_NT_local[((i_c * 8) + 3)] + (A_shared_local[i_c] * B_shared_local[3]));
                  T_batch_matmul_NT_local[((i_c * 8) + 4)] = (T_batch_matmul_NT_local[((i_c * 8) + 4)] + (A_shared_local[i_c] * B_shared_local[4]));
                  T_batch_matmul_NT_local[((i_c * 8) + 5)] = (T_batch_matmul_NT_local[((i_c * 8) + 5)] + (A_shared_local[i_c] * B_shared_local[5]));
                  T_batch_matmul_NT_local[((i_c * 8) + 6)] = (T_batch_matmul_NT_local[((i_c * 8) + 6)] + (A_shared_local[i_c] * B_shared_local[6]));
                  T_batch_matmul_NT_local[((i_c * 8) + 7)] = (T_batch_matmul_NT_local[((i_c * 8) + 7)] + (A_shared_local[i_c] * B_shared_local[7]));
              }
          }
      }
      for (int i_inner_inner = 0; i_inner_inner < 8; ++i_inner_inner) {
          bgemm_0_T_batch_matmul_NT_[((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8))] = T_batch_matmul_NT_local[(i_inner_inner * 8)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 1)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 1)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 2)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 2)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 3)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 3)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 4)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 4)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 5)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 5)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 6)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 6)];
          bgemm_0_T_batch_matmul_NT_[(((((((((int)blockIdx_z_) * 262144) + (((int)blockIdx_y_) * 32768)) + (((int)threadIdx_y_) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx_x_) * 64)) + (((int)threadIdx_x_) * 8)) + 7)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 7)];
      }
  }
  // conv2d_3
  else if ((KernelID_ == 1) && ((int)threadIdx.x < 104))
  {
      float conv2d_nchw_local[16];
      conv2d_nchw_local[0] = 0.F;
      conv2d_nchw_local[2] = 0.F;
      conv2d_nchw_local[4] = 0.F;
      conv2d_nchw_local[6] = 0.F;
      conv2d_nchw_local[8] = 0.F;
      conv2d_nchw_local[10] = 0.F;
      conv2d_nchw_local[12] = 0.F;
      conv2d_nchw_local[14] = 0.F;
      conv2d_nchw_local[1] = 0.F;
      conv2d_nchw_local[3] = 0.F;
      conv2d_nchw_local[5] = 0.F;
      conv2d_nchw_local[7] = 0.F;
      conv2d_nchw_local[9] = 0.F;
      conv2d_nchw_local[11] = 0.F;
      conv2d_nchw_local[13] = 0.F;
      conv2d_nchw_local[15] = 0.F;
      for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
          for (int ry_outer = 0; ry_outer < 3; ++ry_outer) {
              __syncthreads();
              SU_.conv2d_3.pad_temp_shared[((((int)threadIdx_z_) * 26) + (((int)threadIdx_x_) * 2))] = conv2d_3_data_[(((((((((((int)blockIdx_z_) >> 2) * 100352) + (rc_outer * 6272)) + (((int)threadIdx_z_) * 784)) + (((int)blockIdx_y_) * 56)) + (((((int)threadIdx_x_) * 2) / 13) * 28)) + (ry_outer * 28)) + (((int)blockIdx_x_) * 13)) + ((((int)threadIdx_x_) * 2) % 13))];
              SU_.conv2d_3.pad_temp_shared[(((((int)threadIdx_z_) * 26) + (((int)threadIdx_x_) * 2)) + 1)] = conv2d_3_data_[(((((((((((int)blockIdx_z_) >> 2) * 100352) + (rc_outer * 6272)) + (((int)threadIdx_z_) * 784)) + (((int)blockIdx_y_) * 56)) + ((((((int)threadIdx_x_) * 2) + 1) / 13) * 28)) + (ry_outer * 28)) + (((int)blockIdx_x_) * 13)) + (((((int)threadIdx_x_) * 2) + 1) % 13))];
              SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5))] = conv2d_3_kernel_[(((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + (((((int)threadIdx_x_) * 5) >> 3) * 1152)) + (rc_outer * 72)) + (((((int)threadIdx_x_) * 5) & 7) * 9)) + (ry_outer * 3))];
              SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 1)] = conv2d_3_kernel_[(((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 1) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 1) & 7) * 9)) + (ry_outer * 3))];
              SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 2)] = conv2d_3_kernel_[(((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 2) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 2) & 7) * 9)) + (ry_outer * 3))];
              SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 3)] = conv2d_3_kernel_[(((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 3) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 3) & 7) * 9)) + (ry_outer * 3))];
              if (((((((int)threadIdx_x_) * 5) + 4) >> 6) + ((int)threadIdx_z_)) < 8) {
                  if (((int)threadIdx_x_) < 12) {
                      SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 4)] = conv2d_3_kernel_[(((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 4) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 4) & 7) * 9)) + (ry_outer * 3))];
                  }
              }
              __syncthreads();
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[(((int)threadIdx_z_) * 8)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 64)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 128)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 192)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 256)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 320)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 384)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 448)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[(((int)threadIdx_z_) * 8)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 64)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 128)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 192)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 256)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 320)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 384)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 448)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 1)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 65)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 129)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 193)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 257)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 321)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 385)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 449)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 1)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 65)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 129)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 193)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 257)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 321)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 385)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 449)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 2)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 66)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 130)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 194)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 258)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 322)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 386)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 450)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 2)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 66)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 130)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 194)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 258)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 322)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 386)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 450)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 3)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 67)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 131)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 195)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 259)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 323)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 387)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 451)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 3)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 67)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 131)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 195)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 259)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 323)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 387)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 451)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 4)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 68)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 132)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 196)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 260)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 324)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 388)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 452)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 4)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 68)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 132)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 196)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 260)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 324)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 388)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 452)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 5)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 69)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 133)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 197)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 261)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 325)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 389)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 453)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 5)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 69)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 133)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 197)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 261)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 325)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 389)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 453)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 6)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 70)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 134)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 198)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 262)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 326)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 390)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 454)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 6)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 70)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 134)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 198)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 262)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 326)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 390)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 454)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 7)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 71)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 135)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 199)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 263)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 327)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 391)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 455)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 7)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 71)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 135)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 199)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 263)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 327)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 391)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 455)]));
              __syncthreads();
              SU_.conv2d_3.pad_temp_shared[((((int)threadIdx_z_) * 26) + (((int)threadIdx_x_) * 2))] = conv2d_3_data_[((((((((((((int)blockIdx_z_) >> 2) * 100352) + (rc_outer * 6272)) + (((int)threadIdx_z_) * 784)) + (((int)blockIdx_y_) * 56)) + (((((int)threadIdx_x_) * 2) / 13) * 28)) + (ry_outer * 28)) + (((int)blockIdx_x_) * 13)) + ((((int)threadIdx_x_) * 2) % 13)) + 1)];
              SU_.conv2d_3.pad_temp_shared[(((((int)threadIdx_z_) * 26) + (((int)threadIdx_x_) * 2)) + 1)] = conv2d_3_data_[((((((((((((int)blockIdx_z_) >> 2) * 100352) + (rc_outer * 6272)) + (((int)threadIdx_z_) * 784)) + (((int)blockIdx_y_) * 56)) + ((((((int)threadIdx_x_) * 2) + 1) / 13) * 28)) + (ry_outer * 28)) + (((int)blockIdx_x_) * 13)) + (((((int)threadIdx_x_) * 2) + 1) % 13)) + 1)];
              SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5))] = conv2d_3_kernel_[((((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + (((((int)threadIdx_x_) * 5) >> 3) * 1152)) + (rc_outer * 72)) + (((((int)threadIdx_x_) * 5) & 7) * 9)) + (ry_outer * 3)) + 1)];
              SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 1)] = conv2d_3_kernel_[((((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 1) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 1) & 7) * 9)) + (ry_outer * 3)) + 1)];
              SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 2)] = conv2d_3_kernel_[((((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 2) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 2) & 7) * 9)) + (ry_outer * 3)) + 1)];
              SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 3)] = conv2d_3_kernel_[((((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 3) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 3) & 7) * 9)) + (ry_outer * 3)) + 1)];
              if (((((((int)threadIdx_x_) * 5) + 4) >> 6) + ((int)threadIdx_z_)) < 8) {
                  if (((int)threadIdx_x_) < 12) {
                      SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 4)] = conv2d_3_kernel_[((((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 4) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 4) & 7) * 9)) + (ry_outer * 3)) + 1)];
                  }
              }
              __syncthreads();
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[(((int)threadIdx_z_) * 8)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 64)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 128)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 192)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 256)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 320)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 384)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 448)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[(((int)threadIdx_z_) * 8)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 64)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 128)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 192)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 256)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 320)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 384)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 448)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 1)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 65)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 129)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 193)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 257)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 321)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 385)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 449)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 1)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 65)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 129)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 193)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 257)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 321)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 385)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 449)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 2)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 66)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 130)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 194)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 258)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 322)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 386)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 450)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 2)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 66)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 130)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 194)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 258)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 322)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 386)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 450)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 3)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 67)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 131)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 195)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 259)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 323)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 387)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 451)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 3)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 67)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 131)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 195)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 259)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 323)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 387)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 451)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 4)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 68)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 132)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 196)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 260)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 324)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 388)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 452)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 4)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 68)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 132)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 196)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 260)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 324)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 388)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 452)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 5)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 69)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 133)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 197)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 261)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 325)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 389)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 453)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 5)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 69)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 133)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 197)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 261)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 325)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 389)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 453)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 6)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 70)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 134)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 198)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 262)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 326)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 390)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 454)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 6)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 70)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 134)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 198)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 262)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 326)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 390)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 454)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 7)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 71)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 135)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 199)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 263)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 327)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 391)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 455)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 7)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 71)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 135)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 199)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 263)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 327)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 391)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 455)]));
              __syncthreads();
              SU_.conv2d_3.pad_temp_shared[((((int)threadIdx_z_) * 26) + (((int)threadIdx_x_) * 2))] = conv2d_3_data_[((((((((((((int)blockIdx_z_) >> 2) * 100352) + (rc_outer * 6272)) + (((int)threadIdx_z_) * 784)) + (((int)blockIdx_y_) * 56)) + (((((int)threadIdx_x_) * 2) / 13) * 28)) + (ry_outer * 28)) + (((int)blockIdx_x_) * 13)) + ((((int)threadIdx_x_) * 2) % 13)) + 2)];
              SU_.conv2d_3.pad_temp_shared[(((((int)threadIdx_z_) * 26) + (((int)threadIdx_x_) * 2)) + 1)] = conv2d_3_data_[((((((((((((int)blockIdx_z_) >> 2) * 100352) + (rc_outer * 6272)) + (((int)threadIdx_z_) * 784)) + (((int)blockIdx_y_) * 56)) + ((((((int)threadIdx_x_) * 2) + 1) / 13) * 28)) + (ry_outer * 28)) + (((int)blockIdx_x_) * 13)) + (((((int)threadIdx_x_) * 2) + 1) % 13)) + 2)];
              SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5))] = conv2d_3_kernel_[((((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + (((((int)threadIdx_x_) * 5) >> 3) * 1152)) + (rc_outer * 72)) + (((((int)threadIdx_x_) * 5) & 7) * 9)) + (ry_outer * 3)) + 2)];
              SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 1)] = conv2d_3_kernel_[((((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 1) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 1) & 7) * 9)) + (ry_outer * 3)) + 2)];
              SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 2)] = conv2d_3_kernel_[((((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 2) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 2) & 7) * 9)) + (ry_outer * 3)) + 2)];
              SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 3)] = conv2d_3_kernel_[((((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 3) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 3) & 7) * 9)) + (ry_outer * 3)) + 2)];
              if (((((((int)threadIdx_x_) * 5) + 4) >> 6) + ((int)threadIdx_z_)) < 8) {
                  if (((int)threadIdx_x_) < 12) {
                      SU_.conv2d_3.kernel_shared[(((((int)threadIdx_z_) * 64) + (((int)threadIdx_x_) * 5)) + 4)] = conv2d_3_kernel_[((((((((((int)blockIdx_z_) & 3) * 73728) + (((int)threadIdx_z_) * 9216)) + ((((((int)threadIdx_x_) * 5) + 4) >> 3) * 1152)) + (rc_outer * 72)) + ((((((int)threadIdx_x_) * 5) + 4) & 7) * 9)) + (ry_outer * 3)) + 2)];
                  }
              }
              __syncthreads();
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[(((int)threadIdx_z_) * 8)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 64)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 128)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 192)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 256)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 320)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 384)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[((int)threadIdx_x_)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 448)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[(((int)threadIdx_z_) * 8)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 64)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 128)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 192)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 256)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 320)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 384)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 13)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 448)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 1)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 65)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 129)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 193)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 257)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 321)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 385)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 26)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 449)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 1)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 65)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 129)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 193)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 257)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 321)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 385)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 39)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 449)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 2)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 66)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 130)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 194)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 258)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 322)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 386)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 52)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 450)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 2)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 66)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 130)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 194)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 258)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 322)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 386)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 65)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 450)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 3)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 67)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 131)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 195)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 259)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 323)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 387)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 78)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 451)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 3)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 67)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 131)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 195)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 259)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 323)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 387)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 91)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 451)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 4)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 68)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 132)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 196)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 260)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 324)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 388)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 104)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 452)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 4)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 68)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 132)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 196)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 260)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 324)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 388)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 117)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 452)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 5)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 69)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 133)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 197)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 261)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 325)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 389)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 130)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 453)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 5)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 69)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 133)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 197)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 261)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 325)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 389)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 143)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 453)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 6)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 70)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 134)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 198)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 262)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 326)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 390)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 156)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 454)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 6)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 70)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 134)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 198)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 262)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 326)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 390)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 169)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 454)]));
              conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 7)]));
              conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 71)]));
              conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 135)]));
              conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 199)]));
              conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 263)]));
              conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 327)]));
              conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 391)]));
              conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 182)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 455)]));
              conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 7)]));
              conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 71)]));
              conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 135)]));
              conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 199)]));
              conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 263)]));
              conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 327)]));
              conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 391)]));
              conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_3.pad_temp_shared[(((int)threadIdx_x_) + 195)] * SU_.conv2d_3.kernel_shared[((((int)threadIdx_z_) * 8) + 455)]));
          }
      }
      conv2d_3_conv2d_nchw_[(((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_))] = conv2d_nchw_local[0];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 5408)] = conv2d_nchw_local[2];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 10816)] = conv2d_nchw_local[4];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 16224)] = conv2d_nchw_local[6];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 21632)] = conv2d_nchw_local[8];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 27040)] = conv2d_nchw_local[10];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 32448)] = conv2d_nchw_local[12];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 37856)] = conv2d_nchw_local[14];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 26)] = conv2d_nchw_local[1];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 5434)] = conv2d_nchw_local[3];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 10842)] = conv2d_nchw_local[5];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 16250)] = conv2d_nchw_local[7];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 21658)] = conv2d_nchw_local[9];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 27066)] = conv2d_nchw_local[11];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 32474)] = conv2d_nchw_local[13];
      conv2d_3_conv2d_nchw_[((((((((int)blockIdx_z_) * 43264) + (((int)threadIdx_z_) * 676)) + (((int)blockIdx_y_) * 52)) + (((int)blockIdx_x_) * 13)) + ((int)threadIdx_x_)) + 37882)] = conv2d_nchw_local[15];
  }
}
