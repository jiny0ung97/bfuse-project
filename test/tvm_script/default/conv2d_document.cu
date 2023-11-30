
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
extern "C" __global__ void __launch_bounds__(256) default_function_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ W) {
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

