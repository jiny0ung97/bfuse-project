
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
extern "C" __global__ void __launch_bounds__(32) matmul_B128(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ T_matmul_NT) {
  float T_matmul_NT_local[8];
  __shared__ float data_shared[256];
  __shared__ float weight_shared[64];
  T_matmul_NT_local[0] = 0.000000e+00f;
  T_matmul_NT_local[2] = 0.000000e+00f;
  T_matmul_NT_local[4] = 0.000000e+00f;
  T_matmul_NT_local[6] = 0.000000e+00f;
  T_matmul_NT_local[1] = 0.000000e+00f;
  T_matmul_NT_local[3] = 0.000000e+00f;
  T_matmul_NT_local[5] = 0.000000e+00f;
  T_matmul_NT_local[7] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 64; ++k_outer_outer) {
    __syncthreads();
    *(float4*)(data_shared + (((int)threadIdx.x) * 4)) = *(float4*)(data + (((((((int)blockIdx.x) / 125) * 16384) + ((((int)threadIdx.x) >> 1) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 128)) = *(float4*)(data + ((((((((int)blockIdx.x) / 125) * 16384) + ((((int)threadIdx.x) >> 1) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 8192));
    *(float2*)(weight_shared + (((int)threadIdx.x) * 2)) = *(float2*)(weight + (((((((int)blockIdx.x) % 125) * 4096) + ((((int)threadIdx.x) >> 2) * 512)) + (k_outer_outer * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    __syncthreads();
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[((((int)threadIdx.x) >> 1) * 8)] * weight_shared[((((int)threadIdx.x) & 1) * 16)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[((((int)threadIdx.x) >> 1) * 8)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 32)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 128)] * weight_shared[((((int)threadIdx.x) & 1) * 16)]));
    T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 128)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 32)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[((((int)threadIdx.x) >> 1) * 8)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 8)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[((((int)threadIdx.x) >> 1) * 8)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 40)]));
    T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 128)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 8)]));
    T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 128)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 40)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 1)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 1)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 1)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 33)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 129)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 1)]));
    T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 129)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 33)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 1)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 9)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 1)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 41)]));
    T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 129)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 9)]));
    T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 129)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 41)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 2)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 2)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 2)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 34)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 130)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 2)]));
    T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 130)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 34)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 2)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 10)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 2)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 42)]));
    T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 130)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 10)]));
    T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 130)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 42)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 3)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 3)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 3)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 35)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 131)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 3)]));
    T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 131)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 35)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 3)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 11)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 3)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 43)]));
    T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 131)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 11)]));
    T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 131)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 43)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 4)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 4)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 4)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 36)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 132)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 4)]));
    T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 132)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 36)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 4)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 12)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 4)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 44)]));
    T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 132)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 12)]));
    T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 132)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 44)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 5)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 5)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 5)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 37)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 133)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 5)]));
    T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 133)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 37)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 5)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 13)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 5)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 45)]));
    T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 133)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 13)]));
    T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 133)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 45)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 6)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 6)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 6)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 38)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 134)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 6)]));
    T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 134)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 38)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 6)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 14)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 6)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 46)]));
    T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 134)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 14)]));
    T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 134)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 46)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 7)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 7)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 7)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 39)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 135)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 7)]));
    T_matmul_NT_local[6] = (T_matmul_NT_local[6] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 135)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 39)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 7)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 15)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 7)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 47)]));
    T_matmul_NT_local[5] = (T_matmul_NT_local[5] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 135)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 15)]));
    T_matmul_NT_local[7] = (T_matmul_NT_local[7] + (data_shared[(((((int)threadIdx.x) >> 1) * 8) + 135)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 47)]));
  }
  for (int j_inner = 0; j_inner < 2; ++j_inner) {
    T_matmul_NT[((((((((int)blockIdx.x) / 125) * 32000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx.x) % 125) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner)] = T_matmul_NT_local[j_inner];
    T_matmul_NT[(((((((((int)blockIdx.x) / 125) * 32000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx.x) % 125) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner) + 4)] = T_matmul_NT_local[(j_inner + 2)];
    T_matmul_NT[(((((((((int)blockIdx.x) / 125) * 32000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx.x) % 125) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner) + 16000)] = T_matmul_NT_local[(j_inner + 4)];
    T_matmul_NT[(((((((((int)blockIdx.x) / 125) * 32000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx.x) % 125) * 8)) + ((((int)threadIdx.x) & 1) * 2)) + j_inner) + 16004)] = T_matmul_NT_local[(j_inner + 6)];
  }
}

