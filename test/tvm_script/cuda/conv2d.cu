
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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel(float* __restrict__ conv2d_nhwc, float* __restrict__ data, float* __restrict__ kernel) {
  float conv2d_nhwc_local[64];
  __shared__ float pad_temp_shared[1024];
  __shared__ float kernel_shared[3072];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[32] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[33] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[34] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  conv2d_nhwc_local[35] = 0.000000e+00f;
  conv2d_nhwc_local[4] = 0.000000e+00f;
  conv2d_nhwc_local[36] = 0.000000e+00f;
  conv2d_nhwc_local[5] = 0.000000e+00f;
  conv2d_nhwc_local[37] = 0.000000e+00f;
  conv2d_nhwc_local[6] = 0.000000e+00f;
  conv2d_nhwc_local[38] = 0.000000e+00f;
  conv2d_nhwc_local[7] = 0.000000e+00f;
  conv2d_nhwc_local[39] = 0.000000e+00f;
  conv2d_nhwc_local[8] = 0.000000e+00f;
  conv2d_nhwc_local[40] = 0.000000e+00f;
  conv2d_nhwc_local[9] = 0.000000e+00f;
  conv2d_nhwc_local[41] = 0.000000e+00f;
  conv2d_nhwc_local[10] = 0.000000e+00f;
  conv2d_nhwc_local[42] = 0.000000e+00f;
  conv2d_nhwc_local[11] = 0.000000e+00f;
  conv2d_nhwc_local[43] = 0.000000e+00f;
  conv2d_nhwc_local[12] = 0.000000e+00f;
  conv2d_nhwc_local[44] = 0.000000e+00f;
  conv2d_nhwc_local[13] = 0.000000e+00f;
  conv2d_nhwc_local[45] = 0.000000e+00f;
  conv2d_nhwc_local[14] = 0.000000e+00f;
  conv2d_nhwc_local[46] = 0.000000e+00f;
  conv2d_nhwc_local[15] = 0.000000e+00f;
  conv2d_nhwc_local[47] = 0.000000e+00f;
  conv2d_nhwc_local[16] = 0.000000e+00f;
  conv2d_nhwc_local[48] = 0.000000e+00f;
  conv2d_nhwc_local[17] = 0.000000e+00f;
  conv2d_nhwc_local[49] = 0.000000e+00f;
  conv2d_nhwc_local[18] = 0.000000e+00f;
  conv2d_nhwc_local[50] = 0.000000e+00f;
  conv2d_nhwc_local[19] = 0.000000e+00f;
  conv2d_nhwc_local[51] = 0.000000e+00f;
  conv2d_nhwc_local[20] = 0.000000e+00f;
  conv2d_nhwc_local[52] = 0.000000e+00f;
  conv2d_nhwc_local[21] = 0.000000e+00f;
  conv2d_nhwc_local[53] = 0.000000e+00f;
  conv2d_nhwc_local[22] = 0.000000e+00f;
  conv2d_nhwc_local[54] = 0.000000e+00f;
  conv2d_nhwc_local[23] = 0.000000e+00f;
  conv2d_nhwc_local[55] = 0.000000e+00f;
  conv2d_nhwc_local[24] = 0.000000e+00f;
  conv2d_nhwc_local[56] = 0.000000e+00f;
  conv2d_nhwc_local[25] = 0.000000e+00f;
  conv2d_nhwc_local[57] = 0.000000e+00f;
  conv2d_nhwc_local[26] = 0.000000e+00f;
  conv2d_nhwc_local[58] = 0.000000e+00f;
  conv2d_nhwc_local[27] = 0.000000e+00f;
  conv2d_nhwc_local[59] = 0.000000e+00f;
  conv2d_nhwc_local[28] = 0.000000e+00f;
  conv2d_nhwc_local[60] = 0.000000e+00f;
  conv2d_nhwc_local[29] = 0.000000e+00f;
  conv2d_nhwc_local[61] = 0.000000e+00f;
  conv2d_nhwc_local[30] = 0.000000e+00f;
  conv2d_nhwc_local[62] = 0.000000e+00f;
  conv2d_nhwc_local[31] = 0.000000e+00f;
  conv2d_nhwc_local[63] = 0.000000e+00f;
  for (int ry_outer_outer = 0; ry_outer_outer < 3; ++ry_outer_outer) {
    for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
      __syncthreads();
      *(float4*)(pad_temp_shared + (((int)threadIdx.x) * 4)) = (((((1 <= (((((int)blockIdx.x) % 98) / 7) + ry_outer_outer)) && ((((((int)blockIdx.x) % 98) / 7) + ry_outer_outer) < 15)) && (1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) & 7) >> 1)))) && ((((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) & 7) >> 1)) < 15)) ? *(float4*)(data + (((((((((((int)blockIdx.x) / 98) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + (ry_outer_outer * 896)) + ((((int)blockIdx.x) % 98) * 128)) + (((((int)threadIdx.x) & 7) >> 1) * 64)) + (rc_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) - 960)) : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
      *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 512)) = (((((1 <= (((((int)blockIdx.x) % 98) / 7) + ry_outer_outer)) && ((((((int)blockIdx.x) % 98) / 7) + ry_outer_outer) < 15)) && (1 <= (((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) & 7) >> 1)))) && ((((((int)blockIdx.x) % 7) * 2) + ((((int)threadIdx.x) & 7) >> 1)) < 15)) ? *(float4*)(data + (((((((((((int)blockIdx.x) / 98) * 401408) + ((((int)threadIdx.x) >> 3) * 12544)) + (ry_outer_outer * 896)) + ((((int)blockIdx.x) % 98) * 128)) + (((((int)threadIdx.x) & 7) >> 1) * 64)) + (rc_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 199744)) : make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f));
      *(float4*)(kernel_shared + (((int)threadIdx.x) * 4)) = *(float4*)(kernel + (((ry_outer_outer * 24576) + (rc_outer_outer * 1024)) + (((int)threadIdx.x) * 4)));
      *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(kernel + ((((ry_outer_outer * 24576) + (rc_outer_outer * 1024)) + (((int)threadIdx.x) * 4)) + 512));
      *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(kernel + ((((ry_outer_outer * 24576) + (rc_outer_outer * 1024)) + (((int)threadIdx.x) * 4)) + 8192));
      *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(kernel + ((((ry_outer_outer * 24576) + (rc_outer_outer * 1024)) + (((int)threadIdx.x) * 4)) + 8704));
      *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 2048)) = *(float4*)(kernel + ((((ry_outer_outer * 24576) + (rc_outer_outer * 1024)) + (((int)threadIdx.x) * 4)) + 16384));
      *(float4*)(kernel_shared + ((((int)threadIdx.x) * 4) + 2560)) = *(float4*)(kernel + ((((ry_outer_outer * 24576) + (rc_outer_outer * 1024)) + (((int)threadIdx.x) * 4)) + 16896));
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
        conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[(((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[32] = (conv2d_nhwc_local[32] + (pad_temp_shared[(((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[(((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[33] = (conv2d_nhwc_local[33] + (pad_temp_shared[(((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 8)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[34] = (conv2d_nhwc_local[34] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 8)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 8)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[35] = (conv2d_nhwc_local[35] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 8)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 32)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[36] = (conv2d_nhwc_local[36] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 32)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 32)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[37] = (conv2d_nhwc_local[37] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 32)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 40)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[38] = (conv2d_nhwc_local[38] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 40)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 40)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[39] = (conv2d_nhwc_local[39] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 40)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[8] = (conv2d_nhwc_local[8] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 64)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[40] = (conv2d_nhwc_local[40] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 64)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[9] = (conv2d_nhwc_local[9] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 64)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[41] = (conv2d_nhwc_local[41] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 64)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[10] = (conv2d_nhwc_local[10] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 72)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[42] = (conv2d_nhwc_local[42] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 72)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[11] = (conv2d_nhwc_local[11] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 72)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[43] = (conv2d_nhwc_local[43] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 72)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[12] = (conv2d_nhwc_local[12] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 96)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[44] = (conv2d_nhwc_local[44] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 96)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[13] = (conv2d_nhwc_local[13] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 96)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[45] = (conv2d_nhwc_local[45] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 96)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[14] = (conv2d_nhwc_local[14] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 104)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[46] = (conv2d_nhwc_local[46] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 104)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[15] = (conv2d_nhwc_local[15] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 104)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[47] = (conv2d_nhwc_local[47] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 104)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[16] = (conv2d_nhwc_local[16] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 128)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[48] = (conv2d_nhwc_local[48] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 128)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[17] = (conv2d_nhwc_local[17] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 128)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[49] = (conv2d_nhwc_local[49] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 128)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[18] = (conv2d_nhwc_local[18] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 136)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[50] = (conv2d_nhwc_local[50] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 136)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[19] = (conv2d_nhwc_local[19] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 136)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[51] = (conv2d_nhwc_local[51] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 136)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[20] = (conv2d_nhwc_local[20] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 160)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[52] = (conv2d_nhwc_local[52] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 160)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[21] = (conv2d_nhwc_local[21] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 160)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[53] = (conv2d_nhwc_local[53] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 160)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[22] = (conv2d_nhwc_local[22] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 168)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[54] = (conv2d_nhwc_local[54] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 168)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[23] = (conv2d_nhwc_local[23] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 168)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[55] = (conv2d_nhwc_local[55] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 168)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[24] = (conv2d_nhwc_local[24] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 192)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[56] = (conv2d_nhwc_local[56] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 192)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[25] = (conv2d_nhwc_local[25] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 192)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[57] = (conv2d_nhwc_local[57] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 192)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[26] = (conv2d_nhwc_local[26] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 200)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[58] = (conv2d_nhwc_local[58] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 200)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[27] = (conv2d_nhwc_local[27] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 200)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[59] = (conv2d_nhwc_local[59] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 200)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[28] = (conv2d_nhwc_local[28] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 224)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[60] = (conv2d_nhwc_local[60] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 224)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[29] = (conv2d_nhwc_local[29] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 224)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[61] = (conv2d_nhwc_local[61] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 224)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[30] = (conv2d_nhwc_local[30] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 232)] * kernel_shared[((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2))]));
        conv2d_nhwc_local[62] = (conv2d_nhwc_local[62] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 232)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 64)]));
        conv2d_nhwc_local[31] = (conv2d_nhwc_local[31] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 232)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1)]));
        conv2d_nhwc_local[63] = (conv2d_nhwc_local[63] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 232)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 65)]));
        conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 8)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[32] = (conv2d_nhwc_local[32] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 8)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 8)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[33] = (conv2d_nhwc_local[33] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 8)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 16)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[34] = (conv2d_nhwc_local[34] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 16)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 16)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[35] = (conv2d_nhwc_local[35] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 16)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 40)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[36] = (conv2d_nhwc_local[36] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 40)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 40)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[37] = (conv2d_nhwc_local[37] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 40)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 48)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[38] = (conv2d_nhwc_local[38] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 48)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 48)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[39] = (conv2d_nhwc_local[39] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 48)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[8] = (conv2d_nhwc_local[8] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 72)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[40] = (conv2d_nhwc_local[40] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 72)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[9] = (conv2d_nhwc_local[9] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 72)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[41] = (conv2d_nhwc_local[41] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 72)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[10] = (conv2d_nhwc_local[10] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 80)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[42] = (conv2d_nhwc_local[42] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 80)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[11] = (conv2d_nhwc_local[11] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 80)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[43] = (conv2d_nhwc_local[43] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 80)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[12] = (conv2d_nhwc_local[12] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 104)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[44] = (conv2d_nhwc_local[44] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 104)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[13] = (conv2d_nhwc_local[13] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 104)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[45] = (conv2d_nhwc_local[45] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 104)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[14] = (conv2d_nhwc_local[14] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 112)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[46] = (conv2d_nhwc_local[46] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 112)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[15] = (conv2d_nhwc_local[15] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 112)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[47] = (conv2d_nhwc_local[47] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 112)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[16] = (conv2d_nhwc_local[16] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 136)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[48] = (conv2d_nhwc_local[48] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 136)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[17] = (conv2d_nhwc_local[17] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 136)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[49] = (conv2d_nhwc_local[49] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 136)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[18] = (conv2d_nhwc_local[18] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 144)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[50] = (conv2d_nhwc_local[50] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 144)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[19] = (conv2d_nhwc_local[19] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 144)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[51] = (conv2d_nhwc_local[51] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 144)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[20] = (conv2d_nhwc_local[20] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 168)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[52] = (conv2d_nhwc_local[52] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 168)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[21] = (conv2d_nhwc_local[21] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 168)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[53] = (conv2d_nhwc_local[53] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 168)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[22] = (conv2d_nhwc_local[22] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 176)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[54] = (conv2d_nhwc_local[54] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 176)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[23] = (conv2d_nhwc_local[23] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 176)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[55] = (conv2d_nhwc_local[55] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 176)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[24] = (conv2d_nhwc_local[24] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 200)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[56] = (conv2d_nhwc_local[56] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 200)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[25] = (conv2d_nhwc_local[25] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 200)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[57] = (conv2d_nhwc_local[57] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 200)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[26] = (conv2d_nhwc_local[26] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 208)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[58] = (conv2d_nhwc_local[58] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 208)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[27] = (conv2d_nhwc_local[27] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 208)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[59] = (conv2d_nhwc_local[59] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 208)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[28] = (conv2d_nhwc_local[28] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 232)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[60] = (conv2d_nhwc_local[60] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 232)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[29] = (conv2d_nhwc_local[29] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 232)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[61] = (conv2d_nhwc_local[61] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 232)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[30] = (conv2d_nhwc_local[30] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 240)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1024)]));
        conv2d_nhwc_local[62] = (conv2d_nhwc_local[62] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 240)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1088)]));
        conv2d_nhwc_local[31] = (conv2d_nhwc_local[31] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 240)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1025)]));
        conv2d_nhwc_local[63] = (conv2d_nhwc_local[63] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 240)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 1089)]));
        conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 16)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[32] = (conv2d_nhwc_local[32] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 16)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 16)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[33] = (conv2d_nhwc_local[33] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 16)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 24)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[34] = (conv2d_nhwc_local[34] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 24)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 24)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[35] = (conv2d_nhwc_local[35] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 24)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[4] = (conv2d_nhwc_local[4] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 48)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[36] = (conv2d_nhwc_local[36] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 48)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[5] = (conv2d_nhwc_local[5] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 48)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[37] = (conv2d_nhwc_local[37] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 48)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[6] = (conv2d_nhwc_local[6] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 56)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[38] = (conv2d_nhwc_local[38] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 56)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[7] = (conv2d_nhwc_local[7] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 56)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[39] = (conv2d_nhwc_local[39] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 56)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[8] = (conv2d_nhwc_local[8] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 80)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[40] = (conv2d_nhwc_local[40] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 80)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[9] = (conv2d_nhwc_local[9] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 80)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[41] = (conv2d_nhwc_local[41] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 80)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[10] = (conv2d_nhwc_local[10] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 88)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[42] = (conv2d_nhwc_local[42] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 88)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[11] = (conv2d_nhwc_local[11] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 88)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[43] = (conv2d_nhwc_local[43] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 88)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[12] = (conv2d_nhwc_local[12] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 112)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[44] = (conv2d_nhwc_local[44] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 112)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[13] = (conv2d_nhwc_local[13] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 112)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[45] = (conv2d_nhwc_local[45] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 112)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[14] = (conv2d_nhwc_local[14] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 120)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[46] = (conv2d_nhwc_local[46] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 120)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[15] = (conv2d_nhwc_local[15] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 120)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[47] = (conv2d_nhwc_local[47] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 120)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[16] = (conv2d_nhwc_local[16] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 144)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[48] = (conv2d_nhwc_local[48] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 144)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[17] = (conv2d_nhwc_local[17] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 144)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[49] = (conv2d_nhwc_local[49] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 144)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[18] = (conv2d_nhwc_local[18] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 152)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[50] = (conv2d_nhwc_local[50] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 152)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[19] = (conv2d_nhwc_local[19] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 152)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[51] = (conv2d_nhwc_local[51] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 152)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[20] = (conv2d_nhwc_local[20] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 176)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[52] = (conv2d_nhwc_local[52] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 176)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[21] = (conv2d_nhwc_local[21] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 176)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[53] = (conv2d_nhwc_local[53] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 176)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[22] = (conv2d_nhwc_local[22] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 184)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[54] = (conv2d_nhwc_local[54] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 184)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[23] = (conv2d_nhwc_local[23] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 184)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[55] = (conv2d_nhwc_local[55] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 184)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[24] = (conv2d_nhwc_local[24] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 208)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[56] = (conv2d_nhwc_local[56] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 208)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[25] = (conv2d_nhwc_local[25] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 208)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[57] = (conv2d_nhwc_local[57] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 208)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[26] = (conv2d_nhwc_local[26] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 216)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[58] = (conv2d_nhwc_local[58] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 216)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[27] = (conv2d_nhwc_local[27] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 216)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[59] = (conv2d_nhwc_local[59] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 216)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[28] = (conv2d_nhwc_local[28] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 240)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[60] = (conv2d_nhwc_local[60] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 240)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[29] = (conv2d_nhwc_local[29] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 240)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[61] = (conv2d_nhwc_local[61] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 240)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
        conv2d_nhwc_local[30] = (conv2d_nhwc_local[30] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 248)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2048)]));
        conv2d_nhwc_local[62] = (conv2d_nhwc_local[62] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 248)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2112)]));
        conv2d_nhwc_local[31] = (conv2d_nhwc_local[31] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 248)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2049)]));
        conv2d_nhwc_local[63] = (conv2d_nhwc_local[63] + (pad_temp_shared[((((((int)threadIdx.x) >> 5) * 256) + rc_outer_inner) + 248)] * kernel_shared[(((rc_outer_inner * 128) + ((((int)threadIdx.x) & 31) * 2)) + 2113)]));
      }
    }
  }
  for (int nn_inner = 0; nn_inner < 8; ++nn_inner) {
    for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
      for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
        conv2d_nhwc[((((((((((int)blockIdx.x) / 98) * 802816) + ((((int)threadIdx.x) >> 5) * 200704)) + (nn_inner * 25088)) + ((((int)blockIdx.x) % 98) * 256)) + (xx_inner * 128)) + ((((int)threadIdx.x) & 31) * 2)) + ff_inner)] = conv2d_nhwc_local[(((nn_inner * 4) + (xx_inner * 2)) + ff_inner)];
        conv2d_nhwc[(((((((((((int)blockIdx.x) / 98) * 802816) + ((((int)threadIdx.x) >> 5) * 200704)) + (nn_inner * 25088)) + ((((int)blockIdx.x) % 98) * 256)) + (xx_inner * 128)) + ((((int)threadIdx.x) & 31) * 2)) + ff_inner) + 64)] = conv2d_nhwc_local[((((nn_inner * 4) + (xx_inner * 2)) + ff_inner) + 32)];
      }
    }
  }
}

