#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "operation.h"
#include "utils.h"

static bool validation = false;
static size_t T = 0;

static size_t num_iterations = 1;
static size_t fusion_type = 0;
static size_t B[2] = {1, 1};

static const char *operation_type_string[] = {"conv2d", "matmul", "conv2d_matmul", "conv2d_conv2d", "matmul_matmul"};
static const char *operation_fusion_type_string[] = {"parallel", "HFuse", "BFuse", "BFuse++"};

static void print_help(const char *prog_name)
{
  printf(
      "Usage: %s [-vh] [-n num_iterations] [-t fusion_type] C M T\n",
      prog_name);
  printf("Options:\n");
  printf("     -v : validate test.          (default: off)\n");
  printf("     -h : print this page.\n");
  printf("     -n : number of iterations    (default: 1)\n");
  printf("     -t : type of fusion          (default: 0)\n");
  printf("     B1 : batch size of first op  (default: 1)\n");
  printf("     B2 : batch size of second op (default: 1)\n");
  printf("      T : type of operation       (default: 0)\n");
  printf("            0 : conv2d\n");
  printf("            1 : matmul\n");
  printf("            2 : conv2d x matmul\n");
  printf("            3 : conv2d x conv2d\n");
  printf("            4 : matmul x matmul\n");
}

static void parse_opt(int argc, char **argv)
{
  int c;
  while ((c = getopt(argc, argv, "vhn:t:")) != -1)
  {
    switch (c)
    {
    case 'v':
      validation = true;
      break;
    case 'n':
      num_iterations = atoi(optarg);
      break;
    case 't':
      fusion_type = atoi(optarg);
      break;
    case 'h':
    default:
      print_help(argv[0]);
      exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j)
  {
    switch (j)
    {
    case 0:
      B[0] = (size_t)atoi(argv[i]);
      break;
    case 1:
      B[1] = (size_t)atoi(argv[i]);
      break;
    case 2:
      T = (size_t)atoi(argv[i]);
      break;
    default:
      break;
    }
  }

  printf("============= TVM Kernel Benchmark =============\n");
  printf("- Batch sizes of Fisrt Op: %lu\n", B[0]);
  printf("- Batch sizes of Second Op: %lu\n", B[1]);
  printf("- Operation Type: %s\n", operation_type_string[T]);
  printf("- Type of fusion: %s\n", operation_fusion_type_string[fusion_type]);
  printf("- Number of iterations: %lu\n", num_iterations);
}

static void
check_conv2d(int idx, float *I, float *F, float *O)
{
  float *O_ans;
  size_t batch_size;

  if (idx == 0)
    batch_size = B[0];
  else
    batch_size = B[1];

  alloc_tensor(&O_ans, batch_size, 28, 28, 128);
  conv2d(I, F, O_ans);
  
  printf("Validation Result: %s\n", check_matrix(O, O_ans, batch_size, 28, 28, 128) ? "VALID" : "INVALID");

  free(O_ans);
}

static void
check_matmul(int idx, float *I, float *F, float *O)
{
  float *O_ans;
  size_t batch_size;

  if (idx == 0)
    batch_size = B[0];
  else
    batch_size = B[1];

  alloc_tensor(&O_ans, batch_size, 1000, 1, 1);
  matmul(I, F, O_ans);
  
  printf("Validation Result: %s\n", check_matrix(O, O_ans, batch_size, 1000, 1, 1) ? "VALID" : "INVALID");

  free(O_ans);
}

int main(int argc, char **argv)
{
  parse_opt(argc, argv);
  fflush(stdout);

  /* Allocate and initialize tensor on CPU */
  float *I0[2], *F0[2], *O0[2];
  float *I1[2], *F1[2], *O1[2];

  for (int i = 0; i < 2; ++i) {
    alloc_tensor(&I0[i], B[i], 56, 56, 64);
    alloc_tensor(&F0[i], 3, 3, 64, 128);
    alloc_tensor(&O0[i], B[i], 28, 28, 128);

    rand_tensor(I0[i], B[i], 56, 56, 64);
    rand_tensor(F0[i], 3, 3, 64, 128);
  }

  for (int i = 0; i < 2; ++i) {
    alloc_tensor(&I1[i], B[1 - i], 512, 1, 1);
    alloc_tensor(&F1[i], 1000, 512, 1, 1);
    alloc_tensor(&O1[i], B[1 - i], 1000, 1, 1);
    
    rand_tensor(I1[i], B[1 - i], 512, 1, 1);
    rand_tensor(F1[i], 1000, 512, 1, 1);
  }

  /* Initialize Operations */
  initialize(B);

  /* Run few warmup iterations... */
  for (size_t i = 0; i < 3; i++)
  {
    for (int i = 0; i < 2; ++i) zero_tensor(O0[i], B[i], 28, 28, 128);
    for (int i = 0; i < 2; ++i) zero_tensor(O1[i], B[1 - i], 1000, 1, 1);
    switch (T)
    {
    case 0:
      conv2d(I0[0], F0[0], O0[0]);
      break;
    case 1:
      matmul(I1[0], F1[0], O1[0]);
      break;
    case 2:
      if (fusion_type == 0)
        conv2d_matmul_parallel(I0[0], F0[0], O0[0], I1[0], F1[0], O1[0]);
      else
        conv2d_matmul_fuse(fusion_type, I0[0], F0[0], O0[0], I1[0], F1[0], O1[0]);
      break;
    case 3:
      if (fusion_type == 0)
        conv2d_conv2d_parallel(I0[0], F0[0], O0[0], I0[1], F0[1], O0[1]);
      else
        conv2d_conv2d_BFuse(I0[0], F0[0], O0[0], I0[1], F0[1], O0[1]);
      break;
    case 4:
      if (fusion_type == 0)
        matmul_matmul_parallel(I1[0], F1[0], O1[0], I1[1], F1[1], O1[1]);
      else
        matmul_matmul_BFuse(I1[0], F1[0], O1[0], I1[1], F1[1], O1[1]);
      break;
    }
  }

  /* Run convolution for num_iterations */
  printf("\n--------------------- Run Benchmark -----------------------\n");

  double elapsed_time_sum = 0;
  for (size_t i = 0; i < num_iterations; ++i)
  {
    printf("[iter %lu] ", i);
    fflush(stdout);

    for (int i = 0; i < 2; ++i) zero_tensor(O0[i], B[i], 28, 28, 128);
    for (int i = 0; i < 2; ++i) zero_tensor(O1[i], B[1 - i], 1000, 1, 1);
    double elapsed_time_iter = -get_current_time();
    switch (T)
    {
    case 0:
      conv2d(I0[0], F0[0], O0[0]);
      break;
    case 1:
      matmul(I1[0], F1[0], O1[0]);
      break;
    case 2:
      if (fusion_type == 0)
        conv2d_matmul_parallel(I0[0], F0[0], O0[0], I1[0], F1[0], O1[0]);
      else
        conv2d_matmul_fuse(fusion_type, I0[0], F0[0], O0[0], I1[0], F1[0], O1[0]);
      break;
    case 3:
      if (fusion_type == 0)
        conv2d_conv2d_parallel(I0[0], F0[0], O0[0], I0[1], F0[1], O0[1]);
      else
        conv2d_conv2d_BFuse(I0[0], F0[0], O0[0], I0[1], F0[1], O0[1]);
      break;
    case 4:
      if (fusion_type == 0)
        matmul_matmul_parallel(I1[0], F1[0], O1[0], I1[1], F1[1], O1[1]);
      else
        matmul_matmul_BFuse(I1[0], F1[0], O1[0], I1[1], F1[1], O1[1]);
      break;
    }
    elapsed_time_iter += get_current_time();

    printf("%.4f s\n", elapsed_time_iter);
    elapsed_time_sum += elapsed_time_iter;
  }

  if (validation)
  {
    printf("\n----------------------- Validation ------------------------\n");
    switch (T)
    {
    case 0:
      check_conv2d(0, I0[0], F0[0], O0[0]);
      break;
    case 1:
      check_matmul(1, I1[0], F1[0], O1[0]);
      break;
    case 2:
      check_conv2d(0, I0[0], F0[0], O0[0]);
      check_matmul(1, I1[0], F1[0], O1[0]);
      break;
    case 3:
      check_conv2d(0, I0[0], F0[0], O0[0]);
      check_conv2d(1, I0[1], F0[1], O0[1]);
      break;
    case 4:
      check_matmul(1, I1[0], F1[0], O1[0]);
      check_matmul(0, I1[1], F1[1], O1[1]);
      break;
    }
  }

  /* Print performance results */
  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("\n-------------------- Benchmark Summary --------------------\n");
  printf("Avg. time        : %.4f s\n", elapsed_time_avg);
  //   printf("Avg. performance : %.1f GFLOPS\n",
  //          2.0 * ON * OC * OH * OW * C * R * S / elapsed_time_avg / 1e9);

  /* Finalize convolution */
  finalize();

  for (int i = 0; i < 2; ++i) {
    free(I0[i]);
    free(F0[i]);
    free(O0[i]);
  }

  for (int i = 0; i < 2; ++i) {
    free(I1[i]);
    free(F1[i]);
    free(O1[i]);
  }

  printf("\n===========================================================\n");
  return 0;
}
