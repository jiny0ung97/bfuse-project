
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
extern "C" __global__ void __launch_bounds__(32) matmul_B64(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ T_matmul_NT) {
  float T_matmul_NT_local[5];
  __shared__ float data_shared[256];
  __shared__ float weight_shared[160];
  T_matmul_NT_local[0] = 0.000000e+00f;
  T_matmul_NT_local[1] = 0.000000e+00f;
  T_matmul_NT_local[2] = 0.000000e+00f;
  T_matmul_NT_local[3] = 0.000000e+00f;
  T_matmul_NT_local[4] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 32; ++k_outer_outer) {
    __syncthreads();
    *(float2*)(data_shared + (((int)threadIdx.x) * 2)) = *(float2*)(data + (((((((int)blockIdx.x) / 100) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 64)) = *(float2*)(data + ((((((((int)blockIdx.x) / 100) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 2048));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 128)) = *(float2*)(data + ((((((((int)blockIdx.x) / 100) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 4096));
    *(float2*)(data_shared + ((((int)threadIdx.x) * 2) + 192)) = *(float2*)(data + ((((((((int)blockIdx.x) / 100) * 8192) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 16)) + ((((int)threadIdx.x) & 7) * 2)) + 6144));
    weight_shared[((int)threadIdx.x)] = weight[(((((((int)blockIdx.x) % 100) * 5120) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    weight_shared[(((int)threadIdx.x) + 32)] = weight[((((((((int)blockIdx.x) % 100) * 5120) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 1024)];
    weight_shared[(((int)threadIdx.x) + 64)] = weight[((((((((int)blockIdx.x) % 100) * 5120) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 2048)];
    weight_shared[(((int)threadIdx.x) + 96)] = weight[((((((((int)blockIdx.x) % 100) * 5120) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 3072)];
    weight_shared[(((int)threadIdx.x) + 128)] = weight[((((((((int)blockIdx.x) % 100) * 5120) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 4096)];
    __syncthreads();
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[((((int)threadIdx.x) >> 1) * 16)] * weight_shared[((((int)threadIdx.x) & 1) * 16)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[((((int)threadIdx.x) >> 1) * 16)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 32)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[((((int)threadIdx.x) >> 1) * 16)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 64)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[((((int)threadIdx.x) >> 1) * 16)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 96)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[((((int)threadIdx.x) >> 1) * 16)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 128)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 1)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 1)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 1)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 33)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 1)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 65)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 1)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 97)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 1)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 129)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 2)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 2)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 2)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 34)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 2)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 66)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 2)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 98)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 2)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 130)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 3)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 3)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 3)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 35)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 3)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 67)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 3)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 99)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 3)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 131)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 4)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 4)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 4)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 36)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 4)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 68)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 4)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 100)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 4)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 132)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 5)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 5)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 5)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 37)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 5)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 69)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 5)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 101)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 5)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 133)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 6)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 6)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 6)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 38)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 6)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 70)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 6)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 102)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 6)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 134)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 7)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 7)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 7)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 39)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 7)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 71)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 7)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 103)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 7)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 135)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 8)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 8)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 8)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 40)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 8)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 72)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 8)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 104)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 8)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 136)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 9)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 9)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 9)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 41)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 9)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 73)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 9)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 105)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 9)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 137)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 10)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 10)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 10)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 42)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 10)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 74)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 10)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 106)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 10)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 138)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 11)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 11)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 11)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 43)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 11)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 75)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 11)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 107)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 11)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 139)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 12)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 12)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 12)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 44)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 12)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 76)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 12)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 108)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 12)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 140)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 13)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 13)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 13)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 45)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 13)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 77)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 13)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 109)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 13)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 141)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 14)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 14)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 14)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 46)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 14)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 78)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 14)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 110)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 14)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 142)]));
    T_matmul_NT_local[0] = (T_matmul_NT_local[0] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 15)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 15)]));
    T_matmul_NT_local[1] = (T_matmul_NT_local[1] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 15)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 47)]));
    T_matmul_NT_local[2] = (T_matmul_NT_local[2] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 15)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 79)]));
    T_matmul_NT_local[3] = (T_matmul_NT_local[3] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 15)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 111)]));
    T_matmul_NT_local[4] = (T_matmul_NT_local[4] + (data_shared[(((((int)threadIdx.x) >> 1) * 16) + 15)] * weight_shared[(((((int)threadIdx.x) & 1) * 16) + 143)]));
  }
  T_matmul_NT[(((((((int)blockIdx.x) / 100) * 16000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx.x) % 100) * 10)) + (((int)threadIdx.x) & 1))] = T_matmul_NT_local[0];
  T_matmul_NT[((((((((int)blockIdx.x) / 100) * 16000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx.x) % 100) * 10)) + (((int)threadIdx.x) & 1)) + 2)] = T_matmul_NT_local[1];
  T_matmul_NT[((((((((int)blockIdx.x) / 100) * 16000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx.x) % 100) * 10)) + (((int)threadIdx.x) & 1)) + 4)] = T_matmul_NT_local[2];
  T_matmul_NT[((((((((int)blockIdx.x) / 100) * 16000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx.x) % 100) * 10)) + (((int)threadIdx.x) & 1)) + 6)] = T_matmul_NT_local[3];
  T_matmul_NT[((((((((int)blockIdx.x) / 100) * 16000) + ((((int)threadIdx.x) >> 1) * 1000)) + ((((int)blockIdx.x) % 100) * 10)) + (((int)threadIdx.x) & 1)) + 8)] = T_matmul_NT_local[4];
}

