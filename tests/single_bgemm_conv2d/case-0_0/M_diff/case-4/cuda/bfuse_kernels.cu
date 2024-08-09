extern "C" __global__ __launch_bounds__(64) void bgemm_0_conv2d_0_fused_bfuse(float *__restrict bgemm_0_A_, float *__restrict bgemm_0_B_, float *__restrict bgemm_0_T_batch_matmul_NT_, float *__restrict conv2d_0_conv2d_nchw_, float *__restrict conv2d_0_data_, float *__restrict conv2d_0_kernel_)
{
  /*
   * KernelID_ means...
   * 0: bgemm_0
   * 1: conv2d_0
   */
  int gridDim_x_, gridDim_y_, gridDim_z_;
  int blockIdx_x_, blockIdx_y_, blockIdx_z_;
  int blockDim_x_, blockDim_y_, blockDim_z_;
  int threadIdx_x_, threadIdx_y_, threadIdx_z_;
  int NewBlockIdx_;
  int KernelID_;
  
  if (((int)blockIdx.x < 327680) && ((int)blockIdx.x % 5 >= 0) && ((int)blockIdx.x % 5 < 1))
  {
    NewBlockIdx_ = int((int)blockIdx.x / 5) * 1 + ((int)blockIdx.x % 5 - 0);
    KernelID_  = 0;
    gridDim_x_ = 8;
    gridDim_y_ = 8;
    gridDim_z_ = 1024;
    blockDim_x_ = 8;
    blockDim_y_ = 8;
    blockDim_z_ = 1;
  }
  else if (((int)blockIdx.x < 327680) && ((int)blockIdx.x % 5 >= 1) && ((int)blockIdx.x % 5 < 5))
  {
    NewBlockIdx_ = int((int)blockIdx.x / 5) * 4 + ((int)blockIdx.x % 5 - 1);
    KernelID_  = 1;
    gridDim_x_ = 55;
    gridDim_y_ = 11;
    gridDim_z_ = 3072;
    blockDim_x_ = 1;
    blockDim_y_ = 1;
    blockDim_z_ = 16;
  }
  else if ((int)blockIdx.x >= 327680 && (int)blockIdx.x < 327680)
  {
    NewBlockIdx_ = (int)blockIdx.x - 327680 + 65536;
    KernelID_  = 0;
    gridDim_x_ = 8;
    gridDim_y_ = 8;
    gridDim_z_ = 1024;
    blockDim_x_ = 8;
    blockDim_y_ = 8;
    blockDim_z_ = 1;
  }
  else if ((int)blockIdx.x >= 327680 && (int)blockIdx.x < 1924096)
  {
    NewBlockIdx_ = (int)blockIdx.x - 327680 + 262144;
    KernelID_  = 1;
    gridDim_x_ = 55;
    gridDim_y_ = 11;
    gridDim_z_ = 3072;
    blockDim_x_ = 1;
    blockDim_y_ = 1;
    blockDim_z_ = 16;
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
  typedef struct conv2d_0 {
     float pad_temp_shared[18];
     float kernel_shared[128];
  } conv2d_0Ty_;
  typedef union ShrdUnion {
    bgemm_0Ty_ bgemm_0;
    conv2d_0Ty_ conv2d_0;
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
  // conv2d_0
  else if ((KernelID_ == 1) && ((int)threadIdx.x < 16))
  {
      float conv2d_nchw_local[20];
      conv2d_nchw_local[0] = 0.F;
      conv2d_nchw_local[10] = 0.F;
      conv2d_nchw_local[2] = 0.F;
      conv2d_nchw_local[12] = 0.F;
      conv2d_nchw_local[4] = 0.F;
      conv2d_nchw_local[14] = 0.F;
      conv2d_nchw_local[6] = 0.F;
      conv2d_nchw_local[16] = 0.F;
      conv2d_nchw_local[8] = 0.F;
      conv2d_nchw_local[18] = 0.F;
      conv2d_nchw_local[1] = 0.F;
      conv2d_nchw_local[11] = 0.F;
      conv2d_nchw_local[3] = 0.F;
      conv2d_nchw_local[13] = 0.F;
      conv2d_nchw_local[5] = 0.F;
      conv2d_nchw_local[15] = 0.F;
      conv2d_nchw_local[7] = 0.F;
      conv2d_nchw_local[17] = 0.F;
      conv2d_nchw_local[9] = 0.F;
      conv2d_nchw_local[19] = 0.F;
      for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
          __syncthreads();
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[(((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2))];
          }
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[(((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2))];
          }
          SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18))];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 9)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 576)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 585)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1152)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1161)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1728)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1737)];
          __syncthreads();
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          __syncthreads();
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 1)];
          }
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 1)];
          }
          SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 10)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 577)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 586)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1153)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1162)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1729)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1738)];
          __syncthreads();
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          __syncthreads();
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 2)];
          }
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 2)];
          }
          SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 2)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 11)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 578)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 587)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1154)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1163)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1730)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1739)];
          __syncthreads();
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          __syncthreads();
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 112)];
          }
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 112)];
          }
          SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 3)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 12)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 579)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 588)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1155)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1164)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1731)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1740)];
          __syncthreads();
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          __syncthreads();
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 113)];
          }
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 113)];
          }
          SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 4)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 13)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 580)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 589)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1156)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1165)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1732)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1741)];
          __syncthreads();
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          __syncthreads();
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 114)];
          }
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 114)];
          }
          SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 5)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 14)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 581)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 590)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1157)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1166)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1733)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1742)];
          __syncthreads();
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          __syncthreads();
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 224)];
          }
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 224)];
          }
          SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 6)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 15)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 582)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 591)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1158)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1167)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1734)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1743)];
          __syncthreads();
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          __syncthreads();
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 225)];
          }
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 225)];
          }
          SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 7)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 16)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 583)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 592)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1159)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1168)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1735)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1744)];
          __syncthreads();
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          __syncthreads();
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[(((int)threadIdx_z_) * 2)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + (((((int)threadIdx_z_) * 2) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + (((((int)threadIdx_z_) * 2) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 226)];
          }
          if (((int)threadIdx_z_) < 9) {
              SU_.conv2d_0.pad_temp_shared[((((int)threadIdx_z_) * 2) + 1)] = conv2d_0_data_[((((((((((int)blockIdx_z_) / 3) * 802816) + (rc_outer * 25088)) + ((((((int)threadIdx_z_) * 2) + 1) / 9) * 12544)) + (((int)blockIdx_y_) * 1120)) + ((((((int)threadIdx_z_) * 2) + 1) % 9) * 112)) + (((int)blockIdx_x_) * 2)) + 226)];
          }
          SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 8)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 8)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 1)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 17)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 2)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 584)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 3)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 593)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 4)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1160)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 5)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1169)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 6)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1736)];
          SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 8) + 7)] = conv2d_0_kernel_[(((((((int)blockIdx_z_) % 3) * 36864) + (((int)threadIdx_z_) * 2304)) + (rc_outer * 18)) + 1745)];
          __syncthreads();
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[(((int)threadIdx_z_) * 4)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 64)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[0] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[2] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[4] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[6] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 2)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[8] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 66)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 1)]));
          conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 65)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (SU_.conv2d_0.pad_temp_shared[9] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (SU_.conv2d_0.pad_temp_shared[11] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (SU_.conv2d_0.pad_temp_shared[13] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (SU_.conv2d_0.pad_temp_shared[15] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
          conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 3)]));
          conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (SU_.conv2d_0.pad_temp_shared[17] * SU_.conv2d_0.kernel_shared[((((int)threadIdx_z_) * 4) + 67)]));
      }
      conv2d_0_conv2d_nchw_[((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_))] = conv2d_nchw_local[0];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 96800)] = conv2d_nchw_local[10];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 55)] = conv2d_nchw_local[2];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 96855)] = conv2d_nchw_local[12];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 110)] = conv2d_nchw_local[4];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 96910)] = conv2d_nchw_local[14];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 165)] = conv2d_nchw_local[6];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 96965)] = conv2d_nchw_local[16];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 220)] = conv2d_nchw_local[8];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 97020)] = conv2d_nchw_local[18];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 3025)] = conv2d_nchw_local[1];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 99825)] = conv2d_nchw_local[11];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 3080)] = conv2d_nchw_local[3];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 99880)] = conv2d_nchw_local[13];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 3135)] = conv2d_nchw_local[5];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 99935)] = conv2d_nchw_local[15];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 3190)] = conv2d_nchw_local[7];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 99990)] = conv2d_nchw_local[17];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 3245)] = conv2d_nchw_local[9];
      conv2d_0_conv2d_nchw_[(((((((int)blockIdx_z_) * 193600) + (((int)threadIdx_z_) * 6050)) + (((int)blockIdx_y_) * 275)) + ((int)blockIdx_x_)) + 100045)] = conv2d_nchw_local[19];
  }
}
