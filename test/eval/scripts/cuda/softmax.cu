
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
extern "C" __global__ void __launch_bounds__(128) default_function_kernel_2(float* __restrict__ T_softmax_exp, float* __restrict__ T_softmax_maxelem) {
  T_softmax_maxelem[((int)threadIdx.x)] = 0.000000e+00f;
  for (int k = 0; k < 1000; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = (T_softmax_maxelem[((int)threadIdx.x)] + T_softmax_exp[((((int)threadIdx.x) * 1000) + k)]);
  }
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_1(float* __restrict__ T_softmax_exp, float* __restrict__ T_softmax_maxelem, float* __restrict__ data) {
  T_softmax_exp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = __expf((data[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - T_softmax_maxelem[(((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) / 125)]));
}

extern "C" __global__ void __launch_bounds__(1024) default_function_kernel_3(float* __restrict__ T_softmax_exp, float* __restrict__ T_softmax_maxelem, float* __restrict__ T_softmax_norm) {
  T_softmax_norm[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (T_softmax_exp[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / T_softmax_maxelem[(((((int)blockIdx.x) * 128) + (((int)threadIdx.x) >> 3)) / 125)]);
}

extern "C" __global__ void __launch_bounds__(128) default_function_kernel(float* __restrict__ T_softmax_maxelem, float* __restrict__ data) {
  T_softmax_maxelem[((int)threadIdx.x)] = -3.402823e+38f;
  for (int k = 0; k < 1000; ++k) {
    T_softmax_maxelem[((int)threadIdx.x)] = max(T_softmax_maxelem[((int)threadIdx.x)], data[((((int)threadIdx.x) * 1000) + k)]);
  }
}

