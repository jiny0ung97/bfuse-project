
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

extern "C" __global__ void __launch_bounds__(128) bgemm_shared_5120(float* __restrict__ A, float* __restrict__ B, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[32];
  __shared__ float A_shared[4096];
  __shared__ float B_shared[1024];
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
    T_batch_matmul_NT_local[(i_c_outer_inner_init * 8)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 16)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 1)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 17)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 2)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 18)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 3)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 19)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 4)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 20)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 5)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 21)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 6)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 22)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 7)] = 0.000000e+00f;
    T_batch_matmul_NT_local[((i_c_outer_inner_init * 8) + 23)] = 0.000000e+00f;
  }
  for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
    __syncthreads();
    *(float2*)(A_shared + (((int)threadIdx.x) * 2)) = *(float2*)(A + (((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 256)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 8192));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 512)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 16384));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 768)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 24576));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 1024)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 32768));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 1280)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 40960));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 1536)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 49152));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 1792)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 57344));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 2048)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 65536));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 2304)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 73728));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 2560)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 81920));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 2816)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 90112));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 3072)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 98304));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 3328)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 106496));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 3584)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 114688));
    *(float2*)(A_shared + ((((int)threadIdx.x) * 2) + 3840)) = *(float2*)(A + ((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 1024)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 15) * 2)) + 122880));
    B_shared[((int)threadIdx.x)] = B[((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    B_shared[(((int)threadIdx.x) + 128)] = B[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4096)];
    B_shared[(((int)threadIdx.x) + 256)] = B[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 8192)];
    B_shared[(((int)threadIdx.x) + 384)] = B[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 12288)];
    B_shared[(((int)threadIdx.x) + 512)] = B[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 16384)];
    B_shared[(((int)threadIdx.x) + 640)] = B[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 20480)];
    B_shared[(((int)threadIdx.x) + 768)] = B[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 24576)];
    B_shared[(((int)threadIdx.x) + 896)] = B[(((((((((int)blockIdx.x) >> 8) * 1048576) + ((((int)blockIdx.x) & 31) * 32768)) + ((((int)threadIdx.x) >> 5) * 1024)) + (k_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 28672)];
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 32; ++k_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
        T_batch_matmul_NT_local[(i_c_outer_inner * 8)] = (T_batch_matmul_NT_local[(i_c_outer_inner * 8)] + (A_shared[((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 16)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 16)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 1)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 1)] + (A_shared[((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner)] * B_shared[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 17)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 17)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2048)] * B_shared[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 2)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 2)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 18)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 18)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 3)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 3)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 32)] * B_shared[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 19)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 19)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2080)] * B_shared[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 4)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 4)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 20)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 20)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 5)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 5)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 64)] * B_shared[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 21)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 21)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2112)] * B_shared[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 6)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 6)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 22)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 22)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared[(((((int)threadIdx.x) & 15) * 64) + k_outer_inner)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 7)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 7)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 96)] * B_shared[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
        T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 23)] = (T_batch_matmul_NT_local[((i_c_outer_inner * 8) + 23)] + (A_shared[(((((((int)threadIdx.x) >> 4) * 256) + (i_c_outer_inner * 128)) + k_outer_inner) + 2144)] * B_shared[((((((int)threadIdx.x) & 15) * 64) + k_outer_inner) + 32)]));
      }
    }
  }
  for (int i_inner = 0; i_inner < 8; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      T_batch_matmul_NT[(((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + j_inner)] = T_batch_matmul_NT_local[((i_inner * 2) + j_inner)];
      T_batch_matmul_NT[((((((((((int)blockIdx.x) >> 5) * 131072) + ((((int)threadIdx.x) >> 4) * 8192)) + (i_inner * 1024)) + ((((int)blockIdx.x) & 31) * 32)) + ((((int)threadIdx.x) & 15) * 2)) + j_inner) + 65536)] = T_batch_matmul_NT_local[(((i_inner * 2) + j_inner) + 16)];
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) conv2d_shared_2048(float* __restrict__ A, float* __restrict__ B, float* __restrict__ W) {
  float B_local[64];
  __shared__ float Apad_shared[1024];
  __shared__ float W_shared[1024];
  float Apad_shared_local[8];
  float W_shared_local[8];
  for (int ff_c_init = 0; ff_c_init < 4; ++ff_c_init) {
    for (int nn_c_init = 0; nn_c_init < 4; ++nn_c_init) {
      B_local[((ff_c_init * 4) + nn_c_init)] = 0.000000e+00f;
      B_local[(((ff_c_init * 4) + nn_c_init) + 32)] = 0.000000e+00f;
      B_local[(((ff_c_init * 4) + nn_c_init) + 16)] = 0.000000e+00f;
      B_local[(((ff_c_init * 4) + nn_c_init) + 48)] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 32; ++rc_outer) {
    for (int ry = 0; ry < 3; ++ry) {
      for (int rx = 0; rx < 3; ++rx) {
        __syncthreads();
        for (int ax3_inner_outer = 0; ax3_inner_outer < 2; ++ax3_inner_outer) {
          if (((int)threadIdx.x) < 128) {
            *(float4*)(Apad_shared + ((((int)threadIdx.x) * 8) + (ax3_inner_outer * 4))) = (((((1 <= ((((int)blockIdx.x) / 112) + ry)) && (((((int)blockIdx.x) / 112) + ry) < 15)) && (1 <= (((((int)blockIdx.x) % 112) >> 3) + rx))) && ((((((int)blockIdx.x) % 112) >> 3) + rx) < 15)) ? *(float4*)(A + (((((((((ry * 917504) + ((((int)blockIdx.x) >> 3) * 65536)) + (rx * 65536)) + (rc_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + (ax3_inner_outer * 4)) - 983040)) : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
          }
        }
        for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
          if (((int)threadIdx.x) < 128) {
            *(float4*)(W_shared + ((((int)threadIdx.x) * 8) + (ax3_inner_outer_1 * 4))) = *(float4*)(W + (((((((ry * 393216) + (rx * 131072)) + (rc_outer * 4096)) + ((((int)threadIdx.x) >> 4) * 512)) + (((((int)blockIdx.x) & 7) >> 1) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + (ax3_inner_outer_1 * 4)));
          }
        }
        __syncthreads();
        for (int rc_inner = 0; rc_inner < 8; ++rc_inner) {
          for (int ax3 = 0; ax3 < 4; ++ax3) {
            Apad_shared_local[ax3] = Apad_shared[(((rc_inner * 128) + ((((int)threadIdx.x) & 15) * 4)) + ax3)];
            Apad_shared_local[(ax3 + 4)] = Apad_shared[((((rc_inner * 128) + ((((int)threadIdx.x) & 15) * 4)) + ax3) + 64)];
          }
          for (int ax3_1 = 0; ax3_1 < 4; ++ax3_1) {
            W_shared_local[ax3_1] = W_shared[(((rc_inner * 128) + ((((int)threadIdx.x) >> 4) * 4)) + ax3_1)];
            W_shared_local[(ax3_1 + 4)] = W_shared[((((rc_inner * 128) + ((((int)threadIdx.x) >> 4) * 4)) + ax3_1) + 64)];
          }
          for (int ff_c = 0; ff_c < 4; ++ff_c) {
            for (int nn_c = 0; nn_c < 4; ++nn_c) {
              B_local[((ff_c * 4) + nn_c)] = (B_local[((ff_c * 4) + nn_c)] + (Apad_shared_local[nn_c] * W_shared_local[ff_c]));
              B_local[(((ff_c * 4) + nn_c) + 32)] = (B_local[(((ff_c * 4) + nn_c) + 32)] + (Apad_shared_local[nn_c] * W_shared_local[(ff_c + 4)]));
              B_local[(((ff_c * 4) + nn_c) + 16)] = (B_local[(((ff_c * 4) + nn_c) + 16)] + (Apad_shared_local[(nn_c + 4)] * W_shared_local[ff_c]));
              B_local[(((ff_c * 4) + nn_c) + 48)] = (B_local[(((ff_c * 4) + nn_c) + 48)] + (Apad_shared_local[(nn_c + 4)] * W_shared_local[(ff_c + 4)]));
            }
          }
        }
      }
    }
  }
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 4; ++ff_inner_inner_inner) {
    for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 4; ++nn_inner_inner_inner) {
      B[(((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner)] = B_local[((ff_inner_inner_inner * 4) + nn_inner_inner_inner)];
      B[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner) + 16384)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 32)];
      B[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner) + 64)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 16)];
      B[((((((((((int)blockIdx.x) >> 1) * 32768) + ((((int)threadIdx.x) >> 4) * 1024)) + (ff_inner_inner_inner * 256)) + ((((int)blockIdx.x) & 1) * 128)) + ((((int)threadIdx.x) & 15) * 4)) + nn_inner_inner_inner) + 16448)] = B_local[(((ff_inner_inner_inner * 4) + nn_inner_inner_inner) + 48)];
    }
  }
}