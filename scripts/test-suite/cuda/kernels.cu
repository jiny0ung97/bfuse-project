
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
extern "C" __global__ void __launch_bounds__(64) bgemm_0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT);
extern "C" __global__ void __launch_bounds__(64) bgemm_0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[64];
  __shared__ float A_shared[512];
  __shared__ float B_shared[512];
  float A_shared_local[8];
  float B_shared_local[8];
  for (int i_c_init = 0; i_c_init < 8; ++i_c_init) {
    for (int j_c_init = 0; j_c_init < 8; ++j_c_init) {
      T_batch_matmul_NT_local[((i_c_init * 8) + j_c_init)] = 0.000000e+00f;
    }
  }
  for (int k_outer = 0; k_outer < 8; ++k_outer) {
    __syncthreads();
    A_shared[((((int)threadIdx.y) * 64) + ((int)threadIdx.x))] = A[(((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x))];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 8)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 64)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 16)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 128)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 24)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 192)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 32)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 256)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 40)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 320)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 48)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 384)];
    A_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 56)] = A[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.y) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 448)];
    B_shared[((((int)threadIdx.y) * 64) + ((int)threadIdx.x))] = B[(((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x))];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 8)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 64)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 16)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 128)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 24)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 192)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 32)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 256)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 40)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 320)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 48)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 384)];
    B_shared[(((((int)threadIdx.y) * 64) + ((int)threadIdx.x)) + 56)] = B[((((((((int)blockIdx.z) * 32768) + (((int)blockIdx.x) * 4096)) + (((int)threadIdx.y) * 512)) + (k_outer * 8)) + ((int)threadIdx.x)) + 448)];
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      A_shared_local[0] = A_shared[((((int)threadIdx.y) * 64) + k_inner)];
      A_shared_local[1] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 8)];
      A_shared_local[2] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 16)];
      A_shared_local[3] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 24)];
      A_shared_local[4] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 32)];
      A_shared_local[5] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 40)];
      A_shared_local[6] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 48)];
      A_shared_local[7] = A_shared[(((((int)threadIdx.y) * 64) + k_inner) + 56)];
      B_shared_local[0] = B_shared[((((int)threadIdx.x) * 64) + k_inner)];
      B_shared_local[1] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 8)];
      B_shared_local[2] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 16)];
      B_shared_local[3] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 24)];
      B_shared_local[4] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 32)];
      B_shared_local[5] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 40)];
      B_shared_local[6] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 48)];
      B_shared_local[7] = B_shared[(((((int)threadIdx.x) * 64) + k_inner) + 56)];
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
    T_batch_matmul_NT[((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8))] = T_batch_matmul_NT_local[(i_inner_inner * 8)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 1)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 1)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 2)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 2)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 3)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 3)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 4)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 4)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 5)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 5)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 6)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 6)];
    T_batch_matmul_NT[(((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.y) * 32768)) + (((int)threadIdx.y) * 4096)) + (i_inner_inner * 512)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + 7)] = T_batch_matmul_NT_local[((i_inner_inner * 8) + 7)];
  }
}


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
extern "C" __global__ void __launch_bounds__(16) conv2d_0(float* __restrict__ conv2d_nchw, float* __restrict__ data, float* __restrict__ kernel);
extern "C" __global__ void __launch_bounds__(16) conv2d_0(float* __restrict__ conv2d_nchw, float* __restrict__ data, float* __restrict__ kernel) {
  float conv2d_nchw_local[20];
  __shared__ float pad_temp_shared[18];
  __shared__ float kernel_shared[128];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    __syncthreads();
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[(((int)threadIdx.z) * 2)] = data[(((((rc_outer * 25088) + (((((int)threadIdx.z) * 2) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + (((((int)threadIdx.z) * 2) % 9) * 112)) + (((int)blockIdx.x) * 2))];
    }
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[((((int)threadIdx.z) * 2) + 1)] = data[(((((rc_outer * 25088) + ((((((int)threadIdx.z) * 2) + 1) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + ((((((int)threadIdx.z) * 2) + 1) % 9) * 112)) + (((int)blockIdx.x) * 2))];
    }
    kernel_shared[(((int)threadIdx.z) * 8)] = kernel[(((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18))];
    kernel_shared[((((int)threadIdx.z) * 8) + 1)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 9)];
    kernel_shared[((((int)threadIdx.z) * 8) + 2)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 576)];
    kernel_shared[((((int)threadIdx.z) * 8) + 3)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 585)];
    kernel_shared[((((int)threadIdx.z) * 8) + 4)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1152)];
    kernel_shared[((((int)threadIdx.z) * 8) + 5)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1161)];
    kernel_shared[((((int)threadIdx.z) * 8) + 6)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1728)];
    kernel_shared[((((int)threadIdx.z) * 8) + 7)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1737)];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    __syncthreads();
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[(((int)threadIdx.z) * 2)] = data[((((((rc_outer * 25088) + (((((int)threadIdx.z) * 2) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + (((((int)threadIdx.z) * 2) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 1)];
    }
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[((((int)threadIdx.z) * 2) + 1)] = data[((((((rc_outer * 25088) + ((((((int)threadIdx.z) * 2) + 1) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + ((((((int)threadIdx.z) * 2) + 1) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 1)];
    }
    kernel_shared[(((int)threadIdx.z) * 8)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1)];
    kernel_shared[((((int)threadIdx.z) * 8) + 1)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 10)];
    kernel_shared[((((int)threadIdx.z) * 8) + 2)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 577)];
    kernel_shared[((((int)threadIdx.z) * 8) + 3)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 586)];
    kernel_shared[((((int)threadIdx.z) * 8) + 4)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1153)];
    kernel_shared[((((int)threadIdx.z) * 8) + 5)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1162)];
    kernel_shared[((((int)threadIdx.z) * 8) + 6)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1729)];
    kernel_shared[((((int)threadIdx.z) * 8) + 7)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1738)];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    __syncthreads();
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[(((int)threadIdx.z) * 2)] = data[((((((rc_outer * 25088) + (((((int)threadIdx.z) * 2) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + (((((int)threadIdx.z) * 2) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 2)];
    }
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[((((int)threadIdx.z) * 2) + 1)] = data[((((((rc_outer * 25088) + ((((((int)threadIdx.z) * 2) + 1) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + ((((((int)threadIdx.z) * 2) + 1) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 2)];
    }
    kernel_shared[(((int)threadIdx.z) * 8)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 2)];
    kernel_shared[((((int)threadIdx.z) * 8) + 1)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 11)];
    kernel_shared[((((int)threadIdx.z) * 8) + 2)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 578)];
    kernel_shared[((((int)threadIdx.z) * 8) + 3)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 587)];
    kernel_shared[((((int)threadIdx.z) * 8) + 4)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1154)];
    kernel_shared[((((int)threadIdx.z) * 8) + 5)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1163)];
    kernel_shared[((((int)threadIdx.z) * 8) + 6)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1730)];
    kernel_shared[((((int)threadIdx.z) * 8) + 7)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1739)];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    __syncthreads();
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[(((int)threadIdx.z) * 2)] = data[((((((rc_outer * 25088) + (((((int)threadIdx.z) * 2) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + (((((int)threadIdx.z) * 2) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 112)];
    }
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[((((int)threadIdx.z) * 2) + 1)] = data[((((((rc_outer * 25088) + ((((((int)threadIdx.z) * 2) + 1) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + ((((((int)threadIdx.z) * 2) + 1) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 112)];
    }
    kernel_shared[(((int)threadIdx.z) * 8)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 3)];
    kernel_shared[((((int)threadIdx.z) * 8) + 1)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 12)];
    kernel_shared[((((int)threadIdx.z) * 8) + 2)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 579)];
    kernel_shared[((((int)threadIdx.z) * 8) + 3)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 588)];
    kernel_shared[((((int)threadIdx.z) * 8) + 4)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1155)];
    kernel_shared[((((int)threadIdx.z) * 8) + 5)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1164)];
    kernel_shared[((((int)threadIdx.z) * 8) + 6)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1731)];
    kernel_shared[((((int)threadIdx.z) * 8) + 7)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1740)];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    __syncthreads();
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[(((int)threadIdx.z) * 2)] = data[((((((rc_outer * 25088) + (((((int)threadIdx.z) * 2) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + (((((int)threadIdx.z) * 2) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 113)];
    }
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[((((int)threadIdx.z) * 2) + 1)] = data[((((((rc_outer * 25088) + ((((((int)threadIdx.z) * 2) + 1) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + ((((((int)threadIdx.z) * 2) + 1) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 113)];
    }
    kernel_shared[(((int)threadIdx.z) * 8)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 4)];
    kernel_shared[((((int)threadIdx.z) * 8) + 1)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 13)];
    kernel_shared[((((int)threadIdx.z) * 8) + 2)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 580)];
    kernel_shared[((((int)threadIdx.z) * 8) + 3)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 589)];
    kernel_shared[((((int)threadIdx.z) * 8) + 4)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1156)];
    kernel_shared[((((int)threadIdx.z) * 8) + 5)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1165)];
    kernel_shared[((((int)threadIdx.z) * 8) + 6)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1732)];
    kernel_shared[((((int)threadIdx.z) * 8) + 7)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1741)];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    __syncthreads();
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[(((int)threadIdx.z) * 2)] = data[((((((rc_outer * 25088) + (((((int)threadIdx.z) * 2) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + (((((int)threadIdx.z) * 2) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 114)];
    }
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[((((int)threadIdx.z) * 2) + 1)] = data[((((((rc_outer * 25088) + ((((((int)threadIdx.z) * 2) + 1) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + ((((((int)threadIdx.z) * 2) + 1) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 114)];
    }
    kernel_shared[(((int)threadIdx.z) * 8)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 5)];
    kernel_shared[((((int)threadIdx.z) * 8) + 1)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 14)];
    kernel_shared[((((int)threadIdx.z) * 8) + 2)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 581)];
    kernel_shared[((((int)threadIdx.z) * 8) + 3)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 590)];
    kernel_shared[((((int)threadIdx.z) * 8) + 4)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1157)];
    kernel_shared[((((int)threadIdx.z) * 8) + 5)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1166)];
    kernel_shared[((((int)threadIdx.z) * 8) + 6)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1733)];
    kernel_shared[((((int)threadIdx.z) * 8) + 7)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1742)];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    __syncthreads();
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[(((int)threadIdx.z) * 2)] = data[((((((rc_outer * 25088) + (((((int)threadIdx.z) * 2) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + (((((int)threadIdx.z) * 2) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 224)];
    }
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[((((int)threadIdx.z) * 2) + 1)] = data[((((((rc_outer * 25088) + ((((((int)threadIdx.z) * 2) + 1) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + ((((((int)threadIdx.z) * 2) + 1) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 224)];
    }
    kernel_shared[(((int)threadIdx.z) * 8)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 6)];
    kernel_shared[((((int)threadIdx.z) * 8) + 1)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 15)];
    kernel_shared[((((int)threadIdx.z) * 8) + 2)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 582)];
    kernel_shared[((((int)threadIdx.z) * 8) + 3)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 591)];
    kernel_shared[((((int)threadIdx.z) * 8) + 4)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1158)];
    kernel_shared[((((int)threadIdx.z) * 8) + 5)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1167)];
    kernel_shared[((((int)threadIdx.z) * 8) + 6)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1734)];
    kernel_shared[((((int)threadIdx.z) * 8) + 7)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1743)];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    __syncthreads();
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[(((int)threadIdx.z) * 2)] = data[((((((rc_outer * 25088) + (((((int)threadIdx.z) * 2) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + (((((int)threadIdx.z) * 2) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 225)];
    }
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[((((int)threadIdx.z) * 2) + 1)] = data[((((((rc_outer * 25088) + ((((((int)threadIdx.z) * 2) + 1) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + ((((((int)threadIdx.z) * 2) + 1) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 225)];
    }
    kernel_shared[(((int)threadIdx.z) * 8)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 7)];
    kernel_shared[((((int)threadIdx.z) * 8) + 1)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 16)];
    kernel_shared[((((int)threadIdx.z) * 8) + 2)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 583)];
    kernel_shared[((((int)threadIdx.z) * 8) + 3)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 592)];
    kernel_shared[((((int)threadIdx.z) * 8) + 4)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1159)];
    kernel_shared[((((int)threadIdx.z) * 8) + 5)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1168)];
    kernel_shared[((((int)threadIdx.z) * 8) + 6)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1735)];
    kernel_shared[((((int)threadIdx.z) * 8) + 7)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1744)];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    __syncthreads();
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[(((int)threadIdx.z) * 2)] = data[((((((rc_outer * 25088) + (((((int)threadIdx.z) * 2) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + (((((int)threadIdx.z) * 2) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 226)];
    }
    if (((int)threadIdx.z) < 9) {
      pad_temp_shared[((((int)threadIdx.z) * 2) + 1)] = data[((((((rc_outer * 25088) + ((((((int)threadIdx.z) * 2) + 1) / 9) * 12544)) + (((int)blockIdx.y) * 1120)) + ((((((int)threadIdx.z) * 2) + 1) % 9) * 112)) + (((int)blockIdx.x) * 2)) + 226)];
    }
    kernel_shared[(((int)threadIdx.z) * 8)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 8)];
    kernel_shared[((((int)threadIdx.z) * 8) + 1)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 17)];
    kernel_shared[((((int)threadIdx.z) * 8) + 2)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 584)];
    kernel_shared[((((int)threadIdx.z) * 8) + 3)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 593)];
    kernel_shared[((((int)threadIdx.z) * 8) + 4)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1160)];
    kernel_shared[((((int)threadIdx.z) * 8) + 5)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1169)];
    kernel_shared[((((int)threadIdx.z) * 8) + 6)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1736)];
    kernel_shared[((((int)threadIdx.z) * 8) + 7)] = kernel[((((((int)blockIdx.z) * 36864) + (((int)threadIdx.z) * 2304)) + (rc_outer * 18)) + 1745)];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[0] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[2] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[4] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[6] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[8] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[0] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[2] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[4] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[6] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[8] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[9] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[11] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[13] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[15] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[17] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
  }
  conv2d_nchw[((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x))] = conv2d_nchw_local[0];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 96800)] = conv2d_nchw_local[10];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 55)] = conv2d_nchw_local[2];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 96855)] = conv2d_nchw_local[12];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 110)] = conv2d_nchw_local[4];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 96910)] = conv2d_nchw_local[14];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 165)] = conv2d_nchw_local[6];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 96965)] = conv2d_nchw_local[16];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 220)] = conv2d_nchw_local[8];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 97020)] = conv2d_nchw_local[18];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 3025)] = conv2d_nchw_local[1];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 99825)] = conv2d_nchw_local[11];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 3080)] = conv2d_nchw_local[3];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 99880)] = conv2d_nchw_local[13];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 3135)] = conv2d_nchw_local[5];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 99935)] = conv2d_nchw_local[15];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 3190)] = conv2d_nchw_local[7];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 99990)] = conv2d_nchw_local[17];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 3245)] = conv2d_nchw_local[9];
  conv2d_nchw[(((((((int)blockIdx.z) * 193600) + (((int)threadIdx.z) * 6050)) + (((int)blockIdx.y) * 275)) + ((int)blockIdx.x)) + 100045)] = conv2d_nchw_local[19];
}

