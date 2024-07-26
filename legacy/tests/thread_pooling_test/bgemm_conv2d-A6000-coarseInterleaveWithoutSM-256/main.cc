
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "macro.h"
#include "operation.h"
#include "utils.h"
//----------------------------------------------------------------------------------------------------
static bool validation       = false;
static size_t num_iterations = 1;
static size_t T              = 0;
static size_t K1             = 0;
static size_t K2             = 0;

static const char *operation_type_string[] = {"kernel1", "kernel2", "parallel", "hfuse", "bfuse"};
//----------------------------------------------------------------------------------------------------
static void print_help(const char *prog_name)
{
  printf(
      "Usage: %s [-vh] [-n num_iterations] T K1 K2\n",
      prog_name);
  printf("Options:\n");
  printf("     -v : validate test.                (default: off)\n");
  printf("     -h : print this page.\n");
  printf("     -n : number of iterations.         (default: 1)\n");
  printf("      T : type of operation.\n");
  printf("     K1 : idx of kernel1.\n");
  printf("     K2 : idx of kernel2.\n");
}
//----------------------------------------------------------------------------------------------------
static void parse_opt(int argc, char **argv)
{
  int c;
  while ((c = getopt(argc, argv, "vhn:")) != -1)
  {
    switch (c)
    {
    case 'v':
      validation = true;
      break;
    case 'n':
      num_iterations = atoi(optarg);
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
    case 1:
      K1 = (size_t)atoi(argv[i]);
      break;
    case 2:
      K2 = (size_t)atoi(argv[i]);
      break;
    default:
      break;
    }
  }

  // printf("============= Benchmark =============\n");
  // printf("- Number of iterations: %lu\n", num_iterations);
  // printf("- Type of operation: %s\n", operation_type_string[T]);
  // printf("- Idx of Kernel1: %lu\n", K1);
  // printf("- Idx of Kernel2: %lu\n", K2);
}
//----------------------------------------------------------------------------------------------------
static bool
check_kernel1(float *I, float *F, float *O, int *I_shape, int *F_shape, int *O_shape)
{
  float *O_ans;
  int *TmpI_shape, *TmpF_shape, *TmpO_shape;
  void(*func)(float*, float*, float*);
  unsigned int *gridDim, *blockDim;

  ASSIGN_KERNEL1(TmpI_shape, TmpF_shape, TmpO_shape, func, gridDim, blockDim, K1);

  alloc_tensor(&O_ans, O_shape[0], O_shape[1], O_shape[2], O_shape[3]);
  run_kernel1(I, F, O_ans, I_shape, F_shape, O_shape, func, gridDim, blockDim, K1);

  // printf("Validation Result: %s\n", check_matrix(O, O_ans, O_shape[0], O_shape[1], O_shape[2], O_shape[3]) ? "VALID" : "INVALID");

  bool result = check_matrix(O, O_ans, O_shape[0], O_shape[1], O_shape[2], O_shape[3]);
  free(O_ans);

  return result;
}
//----------------------------------------------------------------------------------------------------
static bool
check_kernel2(float *I, float *F, float *O, int *I_shape, int *F_shape, int *O_shape)
{
  float *O_ans;
  int *TmpI_shape, *TmpF_shape, *TmpO_shape;
  void(*func)(float*, float*, float*);
  unsigned int *gridDim, *blockDim;

  ASSIGN_KERNEL2(TmpI_shape, TmpF_shape, TmpO_shape, func, gridDim, blockDim, K2);

  alloc_tensor(&O_ans, O_shape[0], O_shape[1], O_shape[2], O_shape[3]);
  run_kernel2(I, F, O_ans, I_shape, F_shape, O_shape, func, gridDim, blockDim, K2);

  // printf("Validation Result: %s\n", check_matrix(O, O_ans, O_shape[0], O_shape[1], O_shape[2], O_shape[3]) ? "VALID" : "INVALID");

  bool result = check_matrix(O, O_ans, O_shape[0], O_shape[1], O_shape[2], O_shape[3]);
  free(O_ans);

  return result;
}
//----------------------------------------------------------------------------------------------------
static void alloc_and_rand_tensor(float **I, float **F, float **O, int *I_shape, int *F_shape, int *O_shape)
{
  alloc_tensor(I, I_shape[0], I_shape[1], I_shape[2], I_shape[3]);
  alloc_tensor(F, F_shape[0], F_shape[1], F_shape[2], F_shape[3]);
  alloc_tensor(O, O_shape[0], O_shape[1], O_shape[2], O_shape[3]);

  rand_tensor(*I, I_shape[0], I_shape[1], I_shape[2], I_shape[3]);
  rand_tensor(*F, F_shape[0], F_shape[1], F_shape[2], F_shape[3]);
}
//----------------------------------------------------------------------------------------------------
static void zero_tensor(float *O, int *O_shape)
{
  zero_tensor(O, O_shape[0], O_shape[1], O_shape[2], O_shape[3]);
}
//----------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  parse_opt(argc, argv);
  fflush(stdout);

  /* Allocate and initialize tensor on CPU */
  float *I1, *F1, *O1;
  float *I2, *F2, *O2;

  int *I1_shape, *F1_shape, *O1_shape;
  int *I2_shape, *F2_shape, *O2_shape;

  void(*func1)(float*, float*, float*);
  void(*func2)(float*, float*, float*);
  void(*func_fused)(float*, float*, float*, float*, float*, float*);

  unsigned int *gridDim1, *blockDim1;
  unsigned int *gridDim2, *blockDim2;

  switch (T)
  {
  case 0:
  case 1:
  case 2:
    ASSIGN_KERNEL1(I1_shape, F1_shape, O1_shape, func1, gridDim1, blockDim1, K1);
    ASSIGN_KERNEL2(I2_shape, F2_shape, O2_shape, func2, gridDim2, blockDim2, K2);
    break;
  case 3:
    ASSIGN_HFUSE(I1_shape, F1_shape, O1_shape, I2_shape, F2_shape, O2_shape, func_fused, gridDim1, blockDim1, K1, K2);
    break;
  case 4:
    ASSIGN_BFUSE(I1_shape, F1_shape, O1_shape, I2_shape, F2_shape, O2_shape, func_fused, gridDim1, blockDim1, K1, K2);
    break;  
  }

  alloc_and_rand_tensor(&I1, &F1, &O1, I1_shape, F1_shape, O1_shape);
  alloc_and_rand_tensor(&I2, &F2, &O2, I2_shape, F2_shape, O2_shape);

  /* Initialize Operations */
  initialize_kernel1(I1_shape, F1_shape, O1_shape);
  initialize_kernel2(I2_shape, F2_shape, O2_shape);

  /* Run few warmup iterations... */
  // for (size_t i = 0; i < 3; i++)
  // {
  //   zero_tensor(O1, O1_shape);
  //   zero_tensor(O2, O2_shape);

  //   switch (T)
  //   {
  //   case 0:
  //     run_kernel1(I1, F1, O1, I1_shape, F1_shape, O1_shape, func1, gridDim1, blockDim1, K1);
  //     break;
  //   case 1:
  //     run_kernel2(I2, F2, O2, I2_shape, F2_shape, O2_shape, func2, gridDim2, blockDim2, K2);
  //     break;
  //   case 2:
  //     run_parallel(I1, F1, O1, I2, F2, O2, I1_shape, F1_shape, O1_shape, I2_shape, F2_shape, O2_shape, func1, func2, gridDim1, blockDim1, gridDim2, blockDim2, K1, K2);
  //     break;
  //   case 3:
  //   case 4:
  //     run_fuse(I1, F1, O1, I2, F2, O2, I1_shape, F1_shape, O1_shape, I2_shape, F2_shape, O2_shape, func_fused, gridDim1, blockDim1, K1, K2);
  //     break;
  //   default:
  //     break;
  //   }
  // }

  /* Run convolution for num_iterations */
  // printf("\n--------------------- Run Benchmark -----------------------\n");

  double elapsed_time_sum = 0;
  for (size_t i = 0; i < num_iterations; ++i)
  {
    // printf("[iter %lu] ", i);
    fflush(stdout);

    zero_tensor(O1, O1_shape);
    zero_tensor(O2, O2_shape);

    double elapsed_time_iter = -get_current_time();
    switch (T)
    {
    case 0:
      run_kernel1(I1, F1, O1, I1_shape, F1_shape, O1_shape, func1, gridDim1, blockDim1, K1);
      break;
    case 1:
      run_kernel2(I2, F2, O2, I2_shape, F2_shape, O2_shape, func2, gridDim2, blockDim2, K2);
      break;
    case 2:
      run_parallel(I1, F1, O1, I2, F2, O2, I1_shape, F1_shape, O1_shape, I2_shape, F2_shape, O2_shape, func1, func2, gridDim1, blockDim1, gridDim2, blockDim2, K1, K2);
      break;
    case 3:
    case 4:
      run_fuse(I1, F1, O1, I2, F2, O2, I1_shape, F1_shape, O1_shape, I2_shape, F2_shape, O2_shape, func_fused, gridDim1, blockDim1, K1, K2);
      break;
    default:
      break;
    }
    elapsed_time_iter += get_current_time();

    // printf("%.4f s\n", elapsed_time_iter);
    elapsed_time_sum += elapsed_time_iter;
  }


  bool result = true;

  if (validation)
  {
    // printf("\n----------------------- Validation ------------------------\n");
    switch (T)
    {
    case 0:
      result = check_kernel1(I1, F1, O1, I1_shape, F1_shape, O1_shape);
      break;
    case 1:
      result = check_kernel2(I2, F2, O2, I2_shape, F2_shape, O2_shape);
      break;
    case 2:
    case 3:
    case 4:
      result = check_kernel1(I1, F1, O1, I1_shape, F1_shape, O1_shape);
      result = result && check_kernel2(I2, F2, O2, I2_shape, F2_shape, O2_shape);
      break;
    default:
      break;
    }
  }

  /* Print performance results */
  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  // printf("\n-------------------- Benchmark Summary --------------------\n");
  // printf("Avg. time        : %.4f s\n", elapsed_time_avg);
  //   printf("Avg. performance : %.1f GFLOPS\n",
  //          2.0 * ON * OC * OH * OW * C * R * S / elapsed_time_avg / 1e9);

  /* Finalize convolution */
  finalize_kernel2();
  finalize_kernel1();

  free(O2);
  free(F2);
  free(I2);
  free(O1);
  free(F1);
  free(I1);
  
  if (validation) {
    if (!result) {
      // printf("Validation Result: %s\n", "INVALID");
      exit(1);
    }
    else {
      // printf("Validation Result: %s\n", "VALID");
    }
  }

  return 0;
}
//----------------------------------------------------------------------------------------------------
