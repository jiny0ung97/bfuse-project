static int kernel1_I_shape_0[] = {1024, 512, 64, 1};
static int kernel1_F_shape_0[] = {1024, 512, 64, 1};
static int kernel1_O_shape_0[] = {1024, 512, 512, 1};
//------------------------------------------------------------------------------------------
static int kernel2_I_shape_0[] = {1024, 16, 227, 227};
static int kernel2_F_shape_0[] = {64, 16, 3, 3};
static int kernel2_O_shape_0[] = {1024, 64, 57, 57};
//------------------------------------------------------------------------------------------
static unsigned int kernel1_gridDim_0[] = {8, 8, 1024};
static unsigned int kernel1_blockDim_0[] = {8, 8, 1};
//------------------------------------------------------------------------------------------
static unsigned int kernel2_gridDim_0[] = {3, 57, 2048};
static unsigned int kernel2_blockDim_0[] = {19, 1, 8};
//------------------------------------------------------------------------------------------
static unsigned int hfuse_gridDim_0[] = {350208, 1, 1};
static unsigned int hfuse_blockDim_0[] = {224, 1, 1};
//------------------------------------------------------------------------------------------
static unsigned int bfuse_gridDim_0[] = {415744, 1, 1};
static unsigned int bfuse_blockDim_0[] = {152, 1, 1};
//------------------------------------------------------------------------------------------
extern "C" void conv2d_4(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0(float* __restrict, float* __restrict, float* __restrict);
//------------------------------------------------------------------------------------------
extern "C" void bgemm_0_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
//------------------------------------------------------------------------------------------
extern "C" void bgemm_0_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
//------------------------------------------------------------------------------------------
#define ASSIGN_KERNEL1(I_shape, F_shape, O_shape, func, gridDim, blockDim, idx) \
  do \
  { \
    switch (idx) \
    { \
    case 0: \
      I_shape  = kernel1_I_shape_0; \
      F_shape  = kernel1_F_shape_0; \
      O_shape  = kernel1_O_shape_0; \
      func     = bgemm_0; \
      gridDim  = kernel1_gridDim_0; \
      blockDim = kernel1_blockDim_0; \
      break; \
    } \
  } while(0)
//------------------------------------------------------------------------------------------
#define ASSIGN_KERNEL2(I_shape, F_shape, O_shape, func, gridDim, blockDim, idx) \
  do \
  { \
    switch (idx) \
    { \
    case 0: \
      I_shape  = kernel2_I_shape_0; \
      F_shape  = kernel2_F_shape_0; \
      O_shape  = kernel2_O_shape_0; \
      func     = conv2d_4; \
      gridDim  = kernel2_gridDim_0; \
      blockDim = kernel2_blockDim_0; \
      break; \
    } \
  } while(0)
//------------------------------------------------------------------------------------------
#define ASSIGN_HFUSE(I1_shape, F1_shape, O1_shape, I2_shape, F2_shpae, O2_shape, func, gridDim, blockDim, idx1, idx2) \
  do \
  { \
    switch (idx1) \
    { \
    case 0: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_0_conv2d_4_fused_hfuse; \
        gridDim  = hfuse_gridDim_0; \
        blockDim = hfuse_blockDim_0; \
        break; \
      } \
      break; \
    } \
  } while(0)
//------------------------------------------------------------------------------------------
#define ASSIGN_BFUSE(I1_shape, F1_shape, O1_shape, I2_shape, F2_shape, O2_shape, func, gridDim, blockDim, idx1, idx2) \
  do \
  { \
    switch (idx1) \
    { \
    case 0: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_0_conv2d_4_fused_bfuse; \
        gridDim  = bfuse_gridDim_0; \
        blockDim = bfuse_blockDim_0; \
        break; \
      } \
      break; \
    } \
  } while(0)
//------------------------------------------------------------------------------------------
