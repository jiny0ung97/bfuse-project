
#pragma once

#include <cstddef>
//------------------------------------------------------------------------------------------
void initialize_kernel1(int *I_shape, int *F_shape, int *O_shape);
void initialize_kernel2(int *I_shape, int *F_shape, int *O_shape);
//------------------------------------------------------------------------------------------
void finalize_kernel1();
void finalize_kernel2();
//------------------------------------------------------------------------------------------
void run_kernel1(float *I, float *F, float *O, int *I_shape, int *F_shape, int *O_shape,
                 void (*func)(float*, float*, float*), unsigned int *gridDim, unsigned int *blockDim, size_t K);
//------------------------------------------------------------------------------------------
void run_kernel2(float *I, float *F, float *O, int *I_shape, int *F_shape, int *O_shape,
                 void (*func)(float*, float*, float*), unsigned int *gridDim, unsigned int *blockDim, size_t K);
//------------------------------------------------------------------------------------------
void run_parallel(float *I1, float *F1, float *O1, float *I2, float *F2, float *O2,
                  int *I1_shape, int *F1_shape, int *O1_shape, int *I2_shape, int *F2_shape, int *O2_shape,
                  void (*func1)(float*, float*, float*), void (*func2)(float*, float*, float*),
                  unsigned int *gridDim1, unsigned int *blockDim1, unsigned int *gridDim2, unsigned int *blockDim2, size_t K1, size_t K2);
//------------------------------------------------------------------------------------------
void run_fuse(float *I1, float *F1, float *O1, float *I2, float *F2, float *O2,
              int *I1_shape, int *F1_shape, int *O1_shape, int *I2_shape, int *F2_shape, int *O2_shape,
              void (*func)(float*, float*, float*, float*, float*, float*), unsigned int *gridDim, unsigned int *blockDim, size_t K1, size_t K2);
//------------------------------------------------------------------------------------------
