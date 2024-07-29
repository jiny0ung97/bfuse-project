
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "utils.h"

double get_current_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void alloc_tensor(float **m, int N, int C, int H, int W) {
  *m = (float *) aligned_alloc(32, N * C * H * W * sizeof(float));
  if (*m == NULL) {
    printf("Failed to allocate memory for tensor.\n");
    exit(0);
  }
}

void rand_tensor(float *m, int N, int C, int H, int W) {
  int L = N * C * H * W;
  for (int j = 0; j < L; j++) { m[j] = (float) rand() / RAND_MAX - 0.5; }
}

void zero_tensor(float *m, int N, int C, int H, int W) {
  int L = N * C * H * W;
  memset(m, 0, sizeof(float) * L);
}

void print_tensor(float *m, int N, int C, int H, int W) {
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      printf("Batch %d, Channel %d\n", n, c);
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          printf("%+.3f ", m[((n * C + c) * H + h) * W + w]);
        }
        printf("\n");
      }
    }
  }
}

bool
check_matrix(float *O, float *O_ans, int N, int C, int H, int W)
{
  const int ON = N;
  const int OC = C;
  const int OH = H;
  const int OW = W;

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int on = 0; on < ON; ++on) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float o = O[((on * OC + oc) * OH + oh) * OW + ow];
          float o_ans = O_ans[((on * OC + oc) * OH + oh) * OW + ow];
          if (fabsf(o - o_ans) > eps &&
              (o_ans == 0 || fabsf((o - o_ans) / o_ans) > eps)) {
            ++cnt;
            if (cnt <= thr)
              printf(
                  "O[%d][%d][%d][%d] : correct_value = %f, your_value = %f\n",
                  on, oc, oh, ow, o_ans, o);
            if (cnt == thr + 1)
              printf("Too many error, only first %d values are printed.\n",
                     thr);
            is_valid = false;
          }
        }
      }
    }
  }
  return is_valid;
}
