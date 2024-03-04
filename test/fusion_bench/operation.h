#pragma once

#include <cstddef>

void conv2d(float *I, float *F, float *O);
void bgemm(float *I, float *F, float *O);
void softmax(float *I, float *O);

void conv2d_parallel(size_t shared_level,
                     float *I0, float *F0, float *O0,
                     float *I1, float *F1, float *O1);
void conv2d_hfuse(size_t shared_level,
                  float *I0, float *F0, float *O0,
                  float *I1, float *F1, float *O1);
void conv2d_bfuse(size_t shared_level,
                  float *I0, float *F0, float *O0,
                  float *I1, float *F1, float *O1);
void bgemm_parallel(size_t shared_level,
                    float *I0, float *F0, float *O0,
                    float *I1, float *F1, float *O1);
void bgemm_hfuse(size_t shared_level,
                 float *I0, float *F0, float *O0,
                 float *I1, float *F1, float *O1);
void bgemm_bfuse(size_t shared_level,
                 float *I0, float *F0, float *O0,
                 float *I1, float *F1, float *O1);
void softmax_parallel(size_t shared_level,
                      float *I0, float *O0,
                      float *I1, float *O1);
void softmax_hfuse(size_t shared_level,
                   float *I0, float *O0,
                   float *I1, float *O1);
void softmax_bfuse(size_t shared_level,
                   float *I0, float *O0,
                   float *I1, float *O1);
void test(size_t shared_level,
          float *I0, float *F0, float *O0,
          float *I1, float *F1, float *O1);
void test_check(float *I0, float *F0, float *O0,
                float *I1, float *F1, float *O1);

void conv2d_initialize();
void bgemm_initialize();
void softmax_initialize();
void test_initialize();

void conv2d_finalize();
void bgemm_finalize();
void softmax_finalize();
void test_finalize();