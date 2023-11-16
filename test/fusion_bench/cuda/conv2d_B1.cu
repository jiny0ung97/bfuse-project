
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
extern "C" __global__ void __launch_bounds__(112) conv2d_B1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nhwc) {
  float conv2d_nhwc_local[4];
  __shared__ float pad_temp_shared[2320];
  __shared__ float kernel_shared[2304];
  conv2d_nhwc_local[0] = 0.000000e+00f;
  conv2d_nhwc_local[1] = 0.000000e+00f;
  conv2d_nhwc_local[2] = 0.000000e+00f;
  conv2d_nhwc_local[3] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((16 <= ((int)blockIdx.x)) && (1 <= ((((((int)blockIdx.x) & 15) >> 3) * 28) + (((int)threadIdx.x) >> 4)))) ? data[(((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3648)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 112)] = ((16 <= ((int)blockIdx.x)) ? data[(((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3200)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 224)] = ((16 <= ((int)blockIdx.x)) ? data[(((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 2752)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 336)] = ((16 <= ((int)blockIdx.x)) ? data[(((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 2304)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 448)] = (((1 <= (((((int)blockIdx.x) >> 4) * 4) + ((((int)threadIdx.x) + 448) / 464))) && (1 <= ((((((int)blockIdx.x) & 15) >> 3) * 28) + (((((int)threadIdx.x) >> 4) + 28) % 29)))) ? data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 448) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((((int)threadIdx.x) >> 4) + 28) % 29) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3648)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 560)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 560) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3264)];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 672) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 2816)];
    pad_temp_shared[(((int)threadIdx.x) + 784)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 784) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 2368)];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = ((1 <= ((((((int)blockIdx.x) & 15) >> 3) * 28) + (((((int)threadIdx.x) >> 4) + 27) % 29))) ? data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 896) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((((int)threadIdx.x) >> 4) + 27) % 29) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3648)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1008)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 1008) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3328)];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 1120) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 2880)];
    pad_temp_shared[(((int)threadIdx.x) + 1232)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 1232) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 2432)];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = ((1 <= ((((((int)blockIdx.x) & 15) >> 3) * 28) + (((((int)threadIdx.x) >> 4) + 26) % 29))) ? data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 1344) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((((int)threadIdx.x) >> 4) + 26) % 29) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3648)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1456)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 1456) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3392)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 1568) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 2944)];
    pad_temp_shared[(((int)threadIdx.x) + 1680)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 1680) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 2496)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = ((1 <= ((((((int)blockIdx.x) & 15) >> 3) * 28) + (((((int)threadIdx.x) >> 4) + 25) % 29))) ? data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 1792) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((((int)threadIdx.x) >> 4) + 25) % 29) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3648)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1904)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 1904) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3456)];
    pad_temp_shared[(((int)threadIdx.x) + 2016)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 2016) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 3008)];
    pad_temp_shared[(((int)threadIdx.x) + 2128)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 2128) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 2560)];
    if (((int)threadIdx.x) < 80) {
      pad_temp_shared[(((int)threadIdx.x) + 2240)] = data[((((((((((int)blockIdx.x) >> 4) * 14336) + (((((int)threadIdx.x) + 2240) / 464) * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 64)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) - 2112)];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((rc_outer_outer * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 112)] = kernel[(((((rc_outer_outer * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 896)];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[(((((((((int)threadIdx.x) + 224) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((((int)threadIdx.x) >> 4) + 14) & 15) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 336)] = kernel[((((((((((int)threadIdx.x) + 336) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 640)];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[(((((((((int)threadIdx.x) + 448) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((((int)threadIdx.x) >> 4) + 12) & 15) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 560)] = kernel[((((((((((int)threadIdx.x) + 560) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 384)];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[(((((((((int)threadIdx.x) + 672) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((((int)threadIdx.x) >> 4) + 10) & 15) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[((((((((((int)threadIdx.x) + 784) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 128)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((((int)threadIdx.x) + 896) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 1024)];
    kernel_shared[(((int)threadIdx.x) + 1008)] = kernel[(((((((((int)threadIdx.x) + 1008) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((((int)threadIdx.x) >> 4) + 15) & 15) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((((((int)threadIdx.x) + 1120) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 768)];
    kernel_shared[(((int)threadIdx.x) + 1232)] = kernel[(((((((((int)threadIdx.x) + 1232) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((((int)threadIdx.x) >> 4) + 13) & 15) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((((int)threadIdx.x) + 1344) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 512)];
    kernel_shared[(((int)threadIdx.x) + 1456)] = kernel[(((((((((int)threadIdx.x) + 1456) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((((int)threadIdx.x) >> 4) + 11) & 15) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((((((int)threadIdx.x) + 1568) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 256)];
    kernel_shared[(((int)threadIdx.x) + 1680)] = kernel[((((((((((int)threadIdx.x) + 1680) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 1152)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((rc_outer_outer * 2048) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 57344)];
    kernel_shared[(((int)threadIdx.x) + 1904)] = kernel[((((((((((int)threadIdx.x) + 1904) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 896)];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[(((((((((int)threadIdx.x) + 2016) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((((int)threadIdx.x) >> 4) + 14) & 15) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 2128)] = kernel[((((((((((int)threadIdx.x) + 2128) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 640)];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((((((int)threadIdx.x) + 2240) >> 8) * 8192) + (rc_outer_outer * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15)) + 1536)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[(((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4))] * kernel_shared[((rc_outer_inner * 64) + (((int)threadIdx.x) & 15))]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 16)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 2)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 32)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 3)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 48)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 16)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 256)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 17)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 272)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 18)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 288)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 19)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 304)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 32)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 512)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 33)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 528)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 34)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 544)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 35)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 560)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 464)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 768)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 465)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 784)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 466)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 800)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 467)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 816)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 480)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1024)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 481)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1040)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 482)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1056)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 483)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1072)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 496)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1280)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 497)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1296)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 498)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1312)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 499)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1328)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 928)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1536)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 929)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1552)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 930)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1568)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 931)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1584)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 944)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1792)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 945)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1808)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 946)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1824)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 947)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1840)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 960)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2048)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 961)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2064)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 962)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2080)]));
      conv2d_nhwc_local[0] = (conv2d_nhwc_local[0] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 963)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2096)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 32)] * kernel_shared[((rc_outer_inner * 64) + (((int)threadIdx.x) & 15))]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 33)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 16)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 34)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 32)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 35)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 48)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 48)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 256)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 49)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 272)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 50)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 288)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 51)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 304)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 64)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 512)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 65)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 528)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 66)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 544)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 67)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 560)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 496)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 768)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 497)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 784)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 498)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 800)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 499)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 816)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 512)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1024)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 513)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1040)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 514)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1056)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 515)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1072)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 528)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1280)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 529)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1296)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 530)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1312)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 531)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1328)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 960)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1536)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 961)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1552)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 962)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1568)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 963)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1584)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 976)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1792)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 977)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1808)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 978)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1824)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 979)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1840)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 992)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2048)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 993)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2064)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 994)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2080)]));
      conv2d_nhwc_local[1] = (conv2d_nhwc_local[1] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 995)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2096)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 928)] * kernel_shared[((rc_outer_inner * 64) + (((int)threadIdx.x) & 15))]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 929)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 16)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 930)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 32)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 931)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 48)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 944)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 256)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 945)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 272)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 946)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 288)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 947)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 304)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 960)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 512)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 961)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 528)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 962)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 544)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 963)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 560)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1392)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 768)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1393)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 784)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1394)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 800)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1395)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 816)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1408)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1024)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1409)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1040)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1410)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1056)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1411)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1072)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1424)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1280)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1425)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1296)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1426)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1312)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1427)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1328)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1856)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1536)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1857)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1552)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1858)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1568)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1859)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1584)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1872)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1792)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1873)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1808)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1874)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1824)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1875)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1840)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1888)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2048)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1889)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2064)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1890)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2080)]));
      conv2d_nhwc_local[2] = (conv2d_nhwc_local[2] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1891)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2096)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 960)] * kernel_shared[((rc_outer_inner * 64) + (((int)threadIdx.x) & 15))]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 961)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 16)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 962)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 32)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 963)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 48)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 976)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 256)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 977)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 272)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 978)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 288)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 979)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 304)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 992)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 512)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 993)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 528)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 994)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 544)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 995)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 560)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1424)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 768)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1425)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 784)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1426)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 800)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1427)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 816)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1440)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1024)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1441)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1040)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1442)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1056)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1443)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1072)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1456)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1280)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1457)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1296)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1458)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1312)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1459)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1328)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1888)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1536)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1889)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1552)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1890)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1568)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1891)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1584)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1904)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1792)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1905)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1808)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1906)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1824)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1907)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 1840)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1920)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2048)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1921)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2064)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1922)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2080)]));
      conv2d_nhwc_local[3] = (conv2d_nhwc_local[3] + (pad_temp_shared[((((((int)threadIdx.x) >> 4) * 64) + (rc_outer_inner * 4)) + 1923)] * kernel_shared[(((rc_outer_inner * 64) + (((int)threadIdx.x) & 15)) + 2096)]));
    }
  }
  for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
    for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
      conv2d_nhwc[((((((((((int)blockIdx.x) >> 4) * 7168) + (yy_inner * 3584)) + (((((int)blockIdx.x) & 15) >> 3) * 1792)) + ((((int)threadIdx.x) >> 4) * 256)) + (xx_inner * 128)) + ((((int)blockIdx.x) & 7) * 16)) + (((int)threadIdx.x) & 15))] = conv2d_nhwc_local[((yy_inner * 2) + xx_inner)];
    }
  }
}

