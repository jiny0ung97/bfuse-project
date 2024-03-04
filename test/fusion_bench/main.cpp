#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "operation.h"
#include "utils.h"
//----------------------------------------------------------------------------------------------------
static bool validation       = false;
static size_t num_iterations = 1;
static size_t fusion_type    = 0;
static size_t shared_level   = 0;

static size_t T = 0;

static const char *operation_type_string[] = {"conv2d", "bgemm", "softmax", "test"};
static const char *operation_fusion_type_string[] = {"parallel", "HFuse", "BFuse"};
//----------------------------------------------------------------------------------------------------
static void print_help(const char *prog_name)
{
  printf(
      "Usage: %s [-vh] [-n num_iterations] [-t fusion_type] [-s shared_level] T\n",
      prog_name);
  printf("Options:\n");
  printf("     -v : validate test.                (default: off)\n");
  printf("     -h : print this page.\n");
  printf("     -n : number of iterations          (default: 1)\n");
  printf("     -t : type of fusion                (default: 0)\n");
  printf("     -s : shared memory footprint level (default: 0)\n");
  printf("      T : type of operation             (default: 0)\n");
  printf("            0 : conv2d\n");
  printf("            1 : bgemm\n");
  printf("            2 : softmax\n");
  printf("            3 : test\n");
}
//----------------------------------------------------------------------------------------------------
static void parse_opt(int argc, char **argv)
{
  int c;
  while ((c = getopt(argc, argv, "vhn:t:s:")) != -1)
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
    case 's':
      shared_level = atoi(optarg);
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
      T = (size_t)atoi(argv[i]);
      break;
    default:
      break;
    }
  }

  printf("============= TVM Kernel Benchmark =============\n");
  printf("- Number of iterations: %lu\n", num_iterations);
  printf("- Type of fusion: %s\n", operation_fusion_type_string[fusion_type]);
  printf("- Shared memory footprint level: %lu\n", shared_level);
  printf("- Operation Type: %s\n", operation_type_string[T]);
}
//----------------------------------------------------------------------------------------------------
static void
check_test(float *I0, float *F0, float *O0, float *I1, float *F1, float *O1)
{
  float *O0_ans, *O1_ans;
  alloc_tensor(&O0_ans, 32, 64, 56, 56);
  alloc_tensor(&O1_ans, 128, 128, 28, 28);
  test_check(I0, F0, O0_ans, I1, F1, O1_ans);
  printf("Validation Result: %s\n", check_matrix(O0, O0_ans, 32, 64, 56, 56) ? "VALID" : "INVALID");
  printf("Validation Result: %s\n", check_matrix(O1, O1_ans, 128, 128, 28, 28) ? "VALID" : "INVALID");
  free(O1_ans);
  free(O0_ans);
}
//----------------------------------------------------------------------------------------------------
static void
check_conv2d(float *I, float *F, float *O)
{
  float *O_ans;
  alloc_tensor(&O_ans, 14, 14, 512, 256);
  conv2d(I, F, O_ans);
  printf("Validation Result: %s\n", check_matrix(O, O_ans, 14, 14, 512, 256) ? "VALID" : "INVALID");
  free(O_ans);
}
//----------------------------------------------------------------------------------------------------
static void
check_bgemm(float *I, float *F, float *O)
{
  float *O_ans;
  alloc_tensor(&O_ans, 128, 1, 1024, 1024);
  bgemm(I, F, O_ans);
  printf("Validation Result: %s\n", check_matrix(O, O_ans, 128, 1, 1024, 1024) ? "VALID" : "INVALID");
  free(O_ans);
}
//----------------------------------------------------------------------------------------------------
static void
check_softmax(float *I, float *O)
{
  float *O_ans;
  alloc_tensor(&O_ans, 128, 1, 1, 1000);
  softmax(I, O_ans);
  printf("Validation Result: %s\n", check_matrix(O, O_ans, 128, 1, 1, 1000) ? "VALID" : "INVALID");
  free(O_ans);
}
//----------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  parse_opt(argc, argv);
  fflush(stdout);

  /* Allocate and initialize tensor on CPU */
  float *I0, *F0, *O0;
  float *I1, *F1, *O1;

  switch (T)
  {
  case 0:
    alloc_tensor(&I0, 14, 14, 256 ,256);
    alloc_tensor(&F0,  3,  3, 256, 512);
    alloc_tensor(&O0, 14, 14, 512, 256);
    alloc_tensor(&I1, 14, 14, 256 ,256);
    alloc_tensor(&F1,  3,  3, 256, 512);
    alloc_tensor(&O1, 14, 14, 512, 256);

    rand_tensor(I0, 14, 14, 256, 256);
    rand_tensor(F0,  3,  3, 256, 512);
    rand_tensor(I1, 14, 14, 256, 256);
    rand_tensor(F1,  3,  3, 256, 512);
    break;
  case 1:
    alloc_tensor(&I0, 128, 1, 1024, 1024);
    alloc_tensor(&F0, 128, 1, 1024, 1024);
    alloc_tensor(&O0, 128, 1, 1024, 1024);
    alloc_tensor(&I1, 128, 1, 1024, 1024);
    alloc_tensor(&F1, 128, 1, 1024, 1024);
    alloc_tensor(&O1, 128, 1, 1024, 1024);

    rand_tensor(I0, 128, 1, 1024, 1024);
    rand_tensor(F0, 128, 1, 1024, 1024);
    rand_tensor(I1, 128, 1, 1024, 1024);
    rand_tensor(F1, 128, 1, 1024, 1024);
    break;
  case 2:
    alloc_tensor(&I0, 128, 1, 1, 1000);
    alloc_tensor(&O0, 128, 1, 1, 1000);
    alloc_tensor(&I1, 128, 1, 1, 1000);
    alloc_tensor(&O1, 128, 1, 1, 1000);

    rand_tensor(I0, 128, 1, 1, 1000);
    rand_tensor(I1, 128, 1, 1, 1000);
    break;
  case 3:
    alloc_tensor(&I0, 32, 64, 56, 56);
    alloc_tensor(&F0, 32, 64, 3, 3);
    alloc_tensor(&O0, 32, 64, 56, 56);
    alloc_tensor(&I1, 128, 128, 56, 56);
    alloc_tensor(&F1, 128, 1, 3, 3);
    alloc_tensor(&O1, 128, 128, 28, 28);

    rand_tensor(I0, 1, 64, 56, 56);
    rand_tensor(F0, 1, 64, 1, 1);
    rand_tensor(I1, 128, 128, 56, 56);
    rand_tensor(F1, 128, 1, 3, 3);
    break;
  default:
    break;
  }

  /* Initialize Operations */
  switch (T)
  {
  case 0:
    conv2d_initialize();
    break;
  case 1:
    bgemm_initialize();
    break;
  case 2:
    softmax_initialize();
    break;
  case 3:
    test_initialize();
    break;
  }

  /* Run few warmup iterations... */
  for (size_t i = 0; i < 3; i++)
  {
    switch (T)
    {
    case 0:
      zero_tensor(O0, 14, 14, 512, 256);
      zero_tensor(O1, 14, 14, 512, 256);
      break;
    case 1:
      zero_tensor(O0, 128, 1, 1024, 1024);
      zero_tensor(O1, 128, 1, 1024, 1024);
      break;
    case 2:
      zero_tensor(O0, 128, 1, 1, 1000);
      zero_tensor(O1, 128, 1, 1, 1000);
      break;
    case 3:
      zero_tensor(O0, 32, 64, 56, 56);
      zero_tensor(O1, 128, 128, 28, 28);
      break;
    }

    switch (T)
    {
    case 0:
      if (fusion_type == 0)
        conv2d_parallel(shared_level, I0, F0, O0, I1, F1, O1);
      else if (fusion_type == 1)
        conv2d_hfuse(shared_level, I0, F0, O0, I1, F1, O1);
      else if (fusion_type == 2)
        conv2d_bfuse(shared_level, I0, F0, O0, I1, F1, O1);
      break;
    case 1:
      if (fusion_type == 0)
        bgemm_parallel(shared_level, I0, F0, O0, I1, F1, O1);
      else if (fusion_type == 1)
        bgemm_hfuse(shared_level, I0, F0, O0, I1, F1, O1);
      else if (fusion_type == 2)
        bgemm_bfuse(shared_level, I0, F0, O0, I1, F1, O1);
      break;
    case 2:
      if (fusion_type == 0)
        softmax_parallel(shared_level, I0, O0, I1, O1);
      else if (fusion_type == 1)
        softmax_hfuse(shared_level, I0, O0, I1, O1);
      else if (fusion_type == 2)
        softmax_bfuse(shared_level, I0, O0, I1, O1);
      break;
    case 3:
      test(shared_level, I0, F0, O0, I1, F1, O1);
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

    switch (T)
    {
    case 0:
      zero_tensor(O0, 14, 14, 512, 256);
      zero_tensor(O1, 14, 14, 512, 256);
      break;
    case 1:
      zero_tensor(O0, 128, 1, 1024, 1024);
      zero_tensor(O1, 128, 1, 1024, 1024);
      break;
    case 2:
      zero_tensor(O0, 128, 1, 1, 1000);
      zero_tensor(O1, 128, 1, 1, 1000);
      break;
    case 3:
      zero_tensor(O0, 32, 64, 56, 56);
      zero_tensor(O1, 128, 128, 28, 28);
      break;
    }

    double elapsed_time_iter = -get_current_time();
    switch (T)
    {
    case 0:
      if (fusion_type == 0)
        conv2d_parallel(shared_level, I0, F0, O0, I1, F1, O1);
      else if (fusion_type == 1)
        conv2d_hfuse(shared_level, I0, F0, O0, I1, F1, O1);
      else if (fusion_type == 2)
        conv2d_bfuse(shared_level, I0, F0, O0, I1, F1, O1);
      break;
    case 1:
      if (fusion_type == 0)
        bgemm_parallel(shared_level, I0, F0, O0, I1, F1, O1);
      else if (fusion_type == 1)
        bgemm_hfuse(shared_level, I0, F0, O0, I1, F1, O1);
      else if (fusion_type == 2)
        bgemm_bfuse(shared_level, I0, F0, O0, I1, F1, O1);
      break;
    case 2:
      if (fusion_type == 0)
        softmax_parallel(shared_level, I0, O0, I1, O1);
      else if (fusion_type == 1)
        softmax_hfuse(shared_level, I0, O0, I1, O1);
      else if (fusion_type == 2)
        softmax_bfuse(shared_level, I0, O0, I1, O1);
      break;
    case 3:
      test(shared_level, I0, F0, O0, I1, F1, O1);
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
      check_conv2d(I0, F0, O0);
      check_conv2d(I1, F1, O1);
      break;
    case 1:
      check_bgemm(I0, F0, O0);
      check_bgemm(I1, F1, O1);
      break;
    case 2:
      check_softmax(I0, O0);
      check_softmax(I1, O1);
      break;
    case 3:
      check_test(I0, F0, O0, I1, F1, O1);
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
  switch (T)
  {
  case 0:
    conv2d_finalize();
    break;
  case 1:
    bgemm_finalize();
    break;
  case 2:
    softmax_finalize();
    break;
  case 3:
    test_finalize();
  }

  switch (T)
  {
  case 0:
  case 1:
  case 3:
    free(O1);
    free(F1);
    free(I1);
    free(O0);
    free(F0);
    free(I0);
    break;
  case 2:
    free(O1);
    free(I1);
    free(O0);
    free(I0);
  }

  printf("\n===========================================================\n");
  return 0;
}
//----------------------------------------------------------------------------------------------------