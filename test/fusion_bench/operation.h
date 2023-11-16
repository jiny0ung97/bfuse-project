#pragma once

#include <cstddef>

void conv2d(float *I, float *F, float *O);
void matmul(float *I, float *F, float *O);
void conv2d_matmul_parallel(float *I0, float *F0, float *O0,
                            float *I1, float *F1, float *O1);
void conv2d_matmul_fuse(size_t type, float *I0, float *F0, float *O0,
                        float *I1, float *F1, float *O1);
void conv2d_conv2d_parallel(float *I0, float *F0, float *O0,
                            float *I1, float *F1, float *O1);
void conv2d_conv2d_BFuse(float *I0, float *F0, float *O0,
                         float *I1, float *F1, float *O1);
void matmul_matmul_parallel(float *I0, float *F0, float *O0,
                            float *I1, float *F1, float *O1);
void matmul_matmul_BFuse(float *I0, float *F0, float *O0,
                         float *I1, float *F1, float *O1);

void initialize(size_t b[2]);
void finalize();