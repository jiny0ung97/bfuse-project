
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
extern "C" __global__ void __launch_bounds__(8) bgemm(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[1];
  __shared__ float A_shared[8];
  __shared__ float B_shared[64];
  float A_shared_local[1];
  float B_shared_local[1];
  T_batch_matmul_NT_local[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    __syncthreads();
    A_shared[((int)threadIdx.x)] = A[(((((int)blockIdx.z) * 512) + (k_outer * 8)) + ((int)threadIdx.x))];
    // #pragma unroll
    for (int ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      B_shared[((ax1_inner * 8) + ((int)threadIdx.x))] = B[(((((((int)blockIdx.z) * 512000) + (((int)blockIdx.x) * 4096)) + (ax1_inner * 512)) + (k_outer * 8)) + ((int)threadIdx.x))];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 8; ++k_inner) {
      A_shared_local[0] = A_shared[k_inner];
      B_shared_local[0] = B_shared[((((int)threadIdx.x) * 8) + k_inner)];
      T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (A_shared_local[0] * B_shared_local[0]));
    }
  }
  T_batch_matmul_NT[(((((int)blockIdx.z) * 1000) + (((int)blockIdx.x) * 8)) + ((int)threadIdx.x))] = T_batch_matmul_NT_local[0];
}

extern "C" __global__ void __launch_bounds__(58) conv2d(float* __restrict__ conv2d_nchw, float* __restrict__ data, float* __restrict__ kernel) {
  float conv2d_nchw_local[64];
  __shared__ float pad_temp_shared[232];
  __shared__ float kernel_shared[256];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[32] = 0.000000e+00f;
  conv2d_nchw_local[34] = 0.000000e+00f;
  conv2d_nchw_local[36] = 0.000000e+00f;
  conv2d_nchw_local[38] = 0.000000e+00f;
  conv2d_nchw_local[40] = 0.000000e+00f;
  conv2d_nchw_local[42] = 0.000000e+00f;
  conv2d_nchw_local[44] = 0.000000e+00f;
  conv2d_nchw_local[46] = 0.000000e+00f;
  conv2d_nchw_local[48] = 0.000000e+00f;
  conv2d_nchw_local[50] = 0.000000e+00f;
  conv2d_nchw_local[52] = 0.000000e+00f;
  conv2d_nchw_local[54] = 0.000000e+00f;
  conv2d_nchw_local[56] = 0.000000e+00f;
  conv2d_nchw_local[58] = 0.000000e+00f;
  conv2d_nchw_local[60] = 0.000000e+00f;
  conv2d_nchw_local[62] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  conv2d_nchw_local[33] = 0.000000e+00f;
  conv2d_nchw_local[35] = 0.000000e+00f;
  conv2d_nchw_local[37] = 0.000000e+00f;
  conv2d_nchw_local[39] = 0.000000e+00f;
  conv2d_nchw_local[41] = 0.000000e+00f;
  conv2d_nchw_local[43] = 0.000000e+00f;
  conv2d_nchw_local[45] = 0.000000e+00f;
  conv2d_nchw_local[47] = 0.000000e+00f;
  conv2d_nchw_local[49] = 0.000000e+00f;
  conv2d_nchw_local[51] = 0.000000e+00f;
  conv2d_nchw_local[53] = 0.000000e+00f;
  conv2d_nchw_local[55] = 0.000000e+00f;
  conv2d_nchw_local[57] = 0.000000e+00f;
  conv2d_nchw_local[59] = 0.000000e+00f;
  conv2d_nchw_local[61] = 0.000000e+00f;
  conv2d_nchw_local[63] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 16; ++rc_outer) {
    __syncthreads();
    pad_temp_shared[((((int)threadIdx.z) * 116) + (((int)threadIdx.x) * 4))] = (((((1 <= ((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 4) % 58) / 29))) && (((((int)blockIdx.y) * 2) + (((((int)threadIdx.x) * 4) % 58) / 29)) < 57)) && (1 <= ((((int)blockIdx.x) * 29) + ((((int)threadIdx.x) * 4) % 29)))) && (((((int)blockIdx.x) * 29) + ((((int)threadIdx.x) * 4) % 29)) < 57)) ? data[(((((((((((int)blockIdx.z) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + (((((int)threadIdx.x) * 2) / 29) * 3136)) + (((int)blockIdx.y) * 112)) + ((((((int)threadIdx.x) * 4) % 58) / 29) * 56)) + (((int)blockIdx.x) * 29)) + ((((int)threadIdx.x) * 4) % 29)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((((int)threadIdx.z) * 116) + (((int)threadIdx.x) * 4)) + 1)] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 4) + 1) % 58) / 29))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 4) + 1) % 58) / 29)) < 57)) && (1 <= ((((int)blockIdx.x) * 29) + (((((int)threadIdx.x) * 4) + 1) % 29)))) && (((((int)blockIdx.x) * 29) + (((((int)threadIdx.x) * 4) + 1) % 29)) < 57)) ? data[(((((((((((int)blockIdx.z) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + (((((int)threadIdx.x) * 2) / 29) * 3136)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 4) + 1) % 58) / 29) * 56)) + (((int)blockIdx.x) * 29)) + (((((int)threadIdx.x) * 4) + 1) % 29)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((((int)threadIdx.z) * 116) + (((int)threadIdx.x) * 4)) + 2)] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 4) + 2) % 58) / 29))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 4) + 2) % 58) / 29)) < 57)) && (1 <= ((((int)blockIdx.x) * 29) + (((((int)threadIdx.x) * 4) + 2) % 29)))) && (((((int)blockIdx.x) * 29) + (((((int)threadIdx.x) * 4) + 2) % 29)) < 57)) ? data[(((((((((((int)blockIdx.z) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 2) + 1) / 29) * 3136)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 4) + 2) % 58) / 29) * 56)) + (((int)blockIdx.x) * 29)) + (((((int)threadIdx.x) * 4) + 2) % 29)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((((int)threadIdx.z) * 116) + (((int)threadIdx.x) * 4)) + 3)] = (((((1 <= ((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 4) + 3) % 58) / 29))) && (((((int)blockIdx.y) * 2) + ((((((int)threadIdx.x) * 4) + 3) % 58) / 29)) < 57)) && (1 <= ((((int)blockIdx.x) * 29) + (((((int)threadIdx.x) * 4) + 3) % 29)))) && (((((int)blockIdx.x) * 29) + (((((int)threadIdx.x) * 4) + 3) % 29)) < 57)) ? data[(((((((((((int)blockIdx.z) * 200704) + (rc_outer * 12544)) + (((int)threadIdx.z) * 6272)) + ((((((int)threadIdx.x) * 2) + 1) / 29) * 3136)) + (((int)blockIdx.y) * 112)) + (((((((int)threadIdx.x) * 4) + 3) % 58) / 29) * 56)) + (((int)blockIdx.x) * 29)) + (((((int)threadIdx.x) * 4) + 3) % 29)) - 57)] : 0.000000e+00f);
    if ((((((int)threadIdx.x) * 5) >> 7) + ((int)threadIdx.z)) < 2) {
      if (((int)threadIdx.x) < 26) {
        kernel_shared[((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 5))] = kernel[((((((int)threadIdx.z) * 2048) + (((((int)threadIdx.x) * 5) >> 2) * 64)) + (rc_outer * 4)) + (((int)threadIdx.x) & 3))];
      }
    }
    if (((((((int)threadIdx.x) * 5) + 1) >> 7) + ((int)threadIdx.z)) < 2) {
      if (((int)threadIdx.x) < 26) {
        kernel_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 5)) + 1)] = kernel[((((((int)threadIdx.z) * 2048) + ((((((int)threadIdx.x) * 5) + 1) >> 2) * 64)) + (rc_outer * 4)) + ((((int)threadIdx.x) + 1) & 3))];
      }
    }
    if (((((((int)threadIdx.x) * 5) + 2) >> 7) + ((int)threadIdx.z)) < 2) {
      if (((int)threadIdx.x) < 26) {
        kernel_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 5)) + 2)] = kernel[((((((int)threadIdx.z) * 2048) + ((((((int)threadIdx.x) * 5) + 2) >> 2) * 64)) + (rc_outer * 4)) + ((((int)threadIdx.x) + 2) & 3))];
      }
    }
    if (((((((int)threadIdx.x) * 5) + 3) >> 7) + ((int)threadIdx.z)) < 2) {
      if (((int)threadIdx.x) < 25) {
        kernel_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 5)) + 3)] = kernel[((((((int)threadIdx.z) * 2048) + ((((((int)threadIdx.x) * 5) + 3) >> 2) * 64)) + (rc_outer * 4)) + ((((int)threadIdx.x) + 3) & 3))];
      }
    }
    if (((((((int)threadIdx.x) * 5) + 4) >> 7) + ((int)threadIdx.z)) < 2) {
      if (((int)threadIdx.x) < 25) {
        kernel_shared[(((((int)threadIdx.z) * 128) + (((int)threadIdx.x) * 5)) + 4)] = kernel[(((((((int)threadIdx.z) * 2048) + (((((int)threadIdx.x) * 5) >> 2) * 64)) + (rc_outer * 4)) + (((int)threadIdx.x) & 3)) + 64)];
      }
    }
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 8)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 16)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 24)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 32)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 40)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 48)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 56)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 72)]));
    conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 80)]));
    conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 88)]));
    conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 96)]));
    conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 104)]));
    conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 112)]));
    conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 120)]));
    conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 128)]));
    conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 136)]));
    conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 144)]));
    conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 152)]));
    conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 160)]));
    conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 168)]));
    conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 176)]));
    conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 184)]));
    conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 192)]));
    conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 200)]));
    conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 208)]));
    conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 216)]));
    conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 224)]));
    conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 232)]));
    conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 240)]));
    conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[((((int)threadIdx.z) * 4) + 248)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[(((int)threadIdx.z) * 4)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 8)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 16)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 24)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 32)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 40)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 48)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 56)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 64)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 72)]));
    conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 80)]));
    conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 88)]));
    conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 96)]));
    conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 104)]));
    conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 112)]));
    conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 120)]));
    conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 128)]));
    conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 136)]));
    conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 144)]));
    conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 152)]));
    conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 160)]));
    conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 168)]));
    conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 176)]));
    conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 184)]));
    conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 192)]));
    conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 200)]));
    conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 208)]));
    conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 216)]));
    conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 224)]));
    conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 232)]));
    conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 240)]));
    conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (pad_temp_shared[(((int)threadIdx.x) + 29)] * kernel_shared[((((int)threadIdx.z) * 4) + 248)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 9)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 17)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 25)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 33)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 41)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 49)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 57)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 73)]));
    conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 81)]));
    conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 89)]));
    conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 97)]));
    conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 105)]));
    conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 113)]));
    conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 121)]));
    conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 129)]));
    conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 137)]));
    conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 145)]));
    conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 153)]));
    conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 161)]));
    conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 169)]));
    conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 177)]));
    conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 185)]));
    conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 193)]));
    conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 201)]));
    conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 209)]));
    conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 217)]));
    conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 225)]));
    conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 233)]));
    conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 241)]));
    conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (pad_temp_shared[(((int)threadIdx.x) + 58)] * kernel_shared[((((int)threadIdx.z) * 4) + 249)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 1)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 9)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 17)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 25)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 33)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 41)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 49)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 57)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 65)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 73)]));
    conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 81)]));
    conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 89)]));
    conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 97)]));
    conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 105)]));
    conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 113)]));
    conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 121)]));
    conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 129)]));
    conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 137)]));
    conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 145)]));
    conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 153)]));
    conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 161)]));
    conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 169)]));
    conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 177)]));
    conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 185)]));
    conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 193)]));
    conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 201)]));
    conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 209)]));
    conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 217)]));
    conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 225)]));
    conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 233)]));
    conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 241)]));
    conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (pad_temp_shared[(((int)threadIdx.x) + 87)] * kernel_shared[((((int)threadIdx.z) * 4) + 249)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 10)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 18)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 26)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 34)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 42)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 50)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 58)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 74)]));
    conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 82)]));
    conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 90)]));
    conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 98)]));
    conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 106)]));
    conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 114)]));
    conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 122)]));
    conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 130)]));
    conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 138)]));
    conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 146)]));
    conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 154)]));
    conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 162)]));
    conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 170)]));
    conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 178)]));
    conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 186)]));
    conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 194)]));
    conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 202)]));
    conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 210)]));
    conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 218)]));
    conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 226)]));
    conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 234)]));
    conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 242)]));
    conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (pad_temp_shared[(((int)threadIdx.x) + 116)] * kernel_shared[((((int)threadIdx.z) * 4) + 250)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 2)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 10)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 18)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 26)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 34)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 42)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 50)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 58)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 66)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 74)]));
    conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 82)]));
    conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 90)]));
    conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 98)]));
    conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 106)]));
    conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 114)]));
    conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 122)]));
    conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 130)]));
    conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 138)]));
    conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 146)]));
    conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 154)]));
    conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 162)]));
    conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 170)]));
    conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 178)]));
    conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 186)]));
    conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 194)]));
    conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 202)]));
    conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 210)]));
    conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 218)]));
    conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 226)]));
    conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 234)]));
    conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 242)]));
    conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (pad_temp_shared[(((int)threadIdx.x) + 145)] * kernel_shared[((((int)threadIdx.z) * 4) + 250)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 11)]));
    conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 19)]));
    conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 27)]));
    conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 35)]));
    conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 43)]));
    conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 51)]));
    conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 59)]));
    conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 75)]));
    conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 83)]));
    conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 91)]));
    conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 99)]));
    conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 107)]));
    conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 115)]));
    conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 123)]));
    conv2d_nchw_local[32] = (conv2d_nchw_local[32] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 131)]));
    conv2d_nchw_local[34] = (conv2d_nchw_local[34] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 139)]));
    conv2d_nchw_local[36] = (conv2d_nchw_local[36] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 147)]));
    conv2d_nchw_local[38] = (conv2d_nchw_local[38] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 155)]));
    conv2d_nchw_local[40] = (conv2d_nchw_local[40] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 163)]));
    conv2d_nchw_local[42] = (conv2d_nchw_local[42] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 171)]));
    conv2d_nchw_local[44] = (conv2d_nchw_local[44] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 179)]));
    conv2d_nchw_local[46] = (conv2d_nchw_local[46] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 187)]));
    conv2d_nchw_local[48] = (conv2d_nchw_local[48] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 195)]));
    conv2d_nchw_local[50] = (conv2d_nchw_local[50] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 203)]));
    conv2d_nchw_local[52] = (conv2d_nchw_local[52] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 211)]));
    conv2d_nchw_local[54] = (conv2d_nchw_local[54] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 219)]));
    conv2d_nchw_local[56] = (conv2d_nchw_local[56] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 227)]));
    conv2d_nchw_local[58] = (conv2d_nchw_local[58] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 235)]));
    conv2d_nchw_local[60] = (conv2d_nchw_local[60] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 243)]));
    conv2d_nchw_local[62] = (conv2d_nchw_local[62] + (pad_temp_shared[(((int)threadIdx.x) + 174)] * kernel_shared[((((int)threadIdx.z) * 4) + 251)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 3)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 11)]));
    conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 19)]));
    conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 27)]));
    conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 35)]));
    conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 43)]));
    conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 51)]));
    conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 59)]));
    conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 67)]));
    conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 75)]));
    conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 83)]));
    conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 91)]));
    conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 99)]));
    conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 107)]));
    conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 115)]));
    conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 123)]));
    conv2d_nchw_local[33] = (conv2d_nchw_local[33] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 131)]));
    conv2d_nchw_local[35] = (conv2d_nchw_local[35] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 139)]));
    conv2d_nchw_local[37] = (conv2d_nchw_local[37] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 147)]));
    conv2d_nchw_local[39] = (conv2d_nchw_local[39] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 155)]));
    conv2d_nchw_local[41] = (conv2d_nchw_local[41] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 163)]));
    conv2d_nchw_local[43] = (conv2d_nchw_local[43] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 171)]));
    conv2d_nchw_local[45] = (conv2d_nchw_local[45] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 179)]));
    conv2d_nchw_local[47] = (conv2d_nchw_local[47] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 187)]));
    conv2d_nchw_local[49] = (conv2d_nchw_local[49] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 195)]));
    conv2d_nchw_local[51] = (conv2d_nchw_local[51] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 203)]));
    conv2d_nchw_local[53] = (conv2d_nchw_local[53] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 211)]));
    conv2d_nchw_local[55] = (conv2d_nchw_local[55] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 219)]));
    conv2d_nchw_local[57] = (conv2d_nchw_local[57] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 227)]));
    conv2d_nchw_local[59] = (conv2d_nchw_local[59] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 235)]));
    conv2d_nchw_local[61] = (conv2d_nchw_local[61] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 243)]));
    conv2d_nchw_local[63] = (conv2d_nchw_local[63] + (pad_temp_shared[(((int)threadIdx.x) + 203)] * kernel_shared[((((int)threadIdx.z) * 4) + 251)]));
  }
  conv2d_nchw[(((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x))] = conv2d_nchw_local[0];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 6728)] = conv2d_nchw_local[2];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 13456)] = conv2d_nchw_local[4];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 20184)] = conv2d_nchw_local[6];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 26912)] = conv2d_nchw_local[8];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 33640)] = conv2d_nchw_local[10];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 40368)] = conv2d_nchw_local[12];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 47096)] = conv2d_nchw_local[14];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 53824)] = conv2d_nchw_local[16];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 60552)] = conv2d_nchw_local[18];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 67280)] = conv2d_nchw_local[20];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 74008)] = conv2d_nchw_local[22];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 80736)] = conv2d_nchw_local[24];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 87464)] = conv2d_nchw_local[26];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 94192)] = conv2d_nchw_local[28];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 100920)] = conv2d_nchw_local[30];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 107648)] = conv2d_nchw_local[32];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 114376)] = conv2d_nchw_local[34];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 121104)] = conv2d_nchw_local[36];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 127832)] = conv2d_nchw_local[38];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 134560)] = conv2d_nchw_local[40];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 141288)] = conv2d_nchw_local[42];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 148016)] = conv2d_nchw_local[44];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 154744)] = conv2d_nchw_local[46];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 161472)] = conv2d_nchw_local[48];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 168200)] = conv2d_nchw_local[50];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 174928)] = conv2d_nchw_local[52];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 181656)] = conv2d_nchw_local[54];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 188384)] = conv2d_nchw_local[56];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 195112)] = conv2d_nchw_local[58];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 201840)] = conv2d_nchw_local[60];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 208568)] = conv2d_nchw_local[62];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 58)] = conv2d_nchw_local[1];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 6786)] = conv2d_nchw_local[3];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 13514)] = conv2d_nchw_local[5];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 20242)] = conv2d_nchw_local[7];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 26970)] = conv2d_nchw_local[9];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 33698)] = conv2d_nchw_local[11];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 40426)] = conv2d_nchw_local[13];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 47154)] = conv2d_nchw_local[15];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 53882)] = conv2d_nchw_local[17];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 60610)] = conv2d_nchw_local[19];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 67338)] = conv2d_nchw_local[21];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 74066)] = conv2d_nchw_local[23];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 80794)] = conv2d_nchw_local[25];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 87522)] = conv2d_nchw_local[27];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 94250)] = conv2d_nchw_local[29];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 100978)] = conv2d_nchw_local[31];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 107706)] = conv2d_nchw_local[33];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 114434)] = conv2d_nchw_local[35];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 121162)] = conv2d_nchw_local[37];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 127890)] = conv2d_nchw_local[39];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 134618)] = conv2d_nchw_local[41];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 141346)] = conv2d_nchw_local[43];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 148074)] = conv2d_nchw_local[45];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 154802)] = conv2d_nchw_local[47];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 161530)] = conv2d_nchw_local[49];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 168258)] = conv2d_nchw_local[51];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 174986)] = conv2d_nchw_local[53];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 181714)] = conv2d_nchw_local[55];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 188442)] = conv2d_nchw_local[57];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 195170)] = conv2d_nchw_local[59];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 201898)] = conv2d_nchw_local[61];
  conv2d_nchw[((((((((int)blockIdx.z) * 215296) + (((int)threadIdx.z) * 3364)) + (((int)blockIdx.y) * 116)) + (((int)blockIdx.x) * 29)) + ((int)threadIdx.x)) + 208626)] = conv2d_nchw_local[63];
}

