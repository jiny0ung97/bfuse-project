
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
extern "C" __global__ void __launch_bounds__(64, 8) conv2d_small(float* __restrict__ A, float* __restrict__ B, float* __restrict__ W) {
  float B_local[256];
  __shared__ float Apad_shared[2048];
  __shared__ float W_shared[1024];
  float Apad_shared_local[16];
  float W_shared_local[16];
  for (int ff_c_init = 0; ff_c_init < 8; ++ff_c_init) {
    for (int nn_c_init = 0; nn_c_init < 8; ++nn_c_init) {
      B_local[((ff_c_init * 8) + nn_c_init)] = 0.000000e+00f;
      B_local[(((ff_c_init * 8) + nn_c_init) + 128)] = 0.000000e+00f;
      B_local[(((ff_c_init * 8) + nn_c_init) + 64)] = 0.000000e+00f;
      B_local[(((ff_c_init * 8) + nn_c_init) + 192)] = 0.000000e+00f;
    }
  }
  for (int rc_outer = 0; rc_outer < 4; ++rc_outer) {
    __syncthreads();
    for (int ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
      for (int ax3_inner_outer = 0; ax3_inner_outer < 4; ++ax3_inner_outer) {
        *(float4*)(Apad_shared + ((((((int)threadIdx.y) * 256) + (ax2_inner * 128)) + (((int)threadIdx.x) * 16)) + (ax3_inner_outer * 4))) = (((((58 <= ((int)blockIdx.z)) && (((int)blockIdx.z) < 3306)) && (1 <= (((int)blockIdx.z) % 58))) && ((((int)blockIdx.z) % 58) < 57)) ? *(float4*)(A + (((((((((((int)blockIdx.z) / 58) * 458752) + ((((int)blockIdx.z) % 58) * 8192)) + (rc_outer * 2048)) + (((int)threadIdx.y) * 256)) + (ax2_inner * 128)) + (((int)threadIdx.x) * 16)) + (ax3_inner_outer * 4)) - 466944)) : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
      }
    }
    for (int ax2_inner_1 = 0; ax2_inner_1 < 2; ++ax2_inner_1) {
      for (int ax3_inner_outer_1 = 0; ax3_inner_outer_1 < 2; ++ax3_inner_outer_1) {
        *(float4*)(W_shared + ((((((int)threadIdx.y) * 128) + (ax2_inner_1 * 64)) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer_1 * 4))) = *(float4*)(W + (((((rc_outer * 1024) + (((int)threadIdx.y) * 128)) + (ax2_inner_1 * 64)) + (((int)threadIdx.x) * 8)) + (ax3_inner_outer_1 * 4)));
      }
    }
    __syncthreads();
    for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
      for (int ax3 = 0; ax3 < 8; ++ax3) {
        Apad_shared_local[ax3] = Apad_shared[(((rc_inner * 128) + (((int)threadIdx.x) * 8)) + ax3)];
        Apad_shared_local[(ax3 + 8)] = Apad_shared[((((rc_inner * 128) + (((int)threadIdx.x) * 8)) + ax3) + 64)];
      }
      for (int ax3_1 = 0; ax3_1 < 8; ++ax3_1) {
        W_shared_local[ax3_1] = W_shared[(((rc_inner * 64) + (((int)threadIdx.y) * 8)) + ax3_1)];
      }
      for (int ff_c = 0; ff_c < 8; ++ff_c) {
        for (int nn_c = 0; nn_c < 8; ++nn_c) {
          B_local[((ff_c * 8) + nn_c)] = (B_local[((ff_c * 8) + nn_c)] + (Apad_shared_local[nn_c] * W_shared_local[ff_c]));
          B_local[(((ff_c * 8) + nn_c) + 64)] = (B_local[(((ff_c * 8) + nn_c) + 64)] + (Apad_shared_local[(nn_c + 8)] * W_shared_local[ff_c]));
        }
      }
    }
  }
  for (int ff_inner_inner_inner = 0; ff_inner_inner_inner < 8; ++ff_inner_inner_inner) {
    for (int nn_inner_inner_inner = 0; nn_inner_inner_inner < 8; ++nn_inner_inner_inner) {
      B[(((((((int)blockIdx.z) * 8192) + (((int)threadIdx.y) * 1024)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx.x) * 8)) + nn_inner_inner_inner)] = B_local[((ff_inner_inner_inner * 8) + nn_inner_inner_inner)];
      B[((((((((int)blockIdx.z) * 8192) + (((int)threadIdx.y) * 1024)) + (ff_inner_inner_inner * 128)) + (((int)threadIdx.x) * 8)) + nn_inner_inner_inner) + 64)] = B_local[(((ff_inner_inner_inner * 8) + nn_inner_inner_inner) + 64)];
    }
  }
}

