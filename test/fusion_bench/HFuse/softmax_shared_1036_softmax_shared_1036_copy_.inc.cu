 __global__ __launch_bounds__(128, 8) void softmax_shared_1036_softmax_shared_1036_copy_fused_kernel_vfuse_lb_idx_0(float *__restrict T_softmax_norm0, float *__restrict data1, float *__restrict T_softmax_norm22, float *__restrict data23)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float normal_reduce_temp02[1];
    static float red_result3[257] __attribute__((shared));
    static float T_softmax_maxelem4[257] __attribute__((shared));
    float normal_reduce_temp0_15[1];
    static float red_result_16[257] __attribute__((shared));
    static float T_softmax_expsum7[257] __attribute__((shared));
    normal_reduce_temp02[0] = -3.40282306E+38F;
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0))]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 128)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 256)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 384)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 512)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 640)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 768)]);
    if (((int)threadIdx_x_0) < 104) {
        normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 896)]);
    }
    float red_buf08[1];
    unsigned int mask9[1];
    float t010[1];
    float red_buf0_111[1];
    unsigned int mask_112[1];
    float t0_113[1];
    static float red_buf_staging14[4] __attribute__((shared));
    red_buf0_111[0] = normal_reduce_temp02[0];
    mask_112[0] = __activemask();
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 16, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 8, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 4, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 2, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 1, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    if ((((int)threadIdx_x_0) % 32) == 0) {
        red_buf_staging14[(((int)threadIdx_x_0) >> 5)] = red_buf0_111[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) < 4) {
        red_buf08[0] = red_buf_staging14[((int)threadIdx_x_0)];
    }
    mask9[0] = (__activemask() & (unsigned int)15);
    t010[0] = __shfl_down_sync(mask9[0], red_buf08[0], 2, 32);
    red_buf08[0] = max(red_buf08[0], t010[0]);
    t010[0] = __shfl_down_sync(mask9[0], red_buf08[0], 1, 32);
    red_buf08[0] = max(red_buf08[0], t010[0]);
    if (((int)threadIdx_x_0) == 0) {
        ((volatile float *)red_result3)[0] = red_buf08[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) == 0) {
        T_softmax_maxelem4[0] = ((volatile float *)red_result3)[0];
    }
    normal_reduce_temp0_15[0] = 0.F;
    asm ("bar.sync 1,128;");
    ;
    for (int k_outer = 0; k_outer < 8; ++k_outer) {
        if (((k_outer * 16) + (((int)threadIdx_x_0) >> 3)) < 125) {
            normal_reduce_temp0_15[0] = (normal_reduce_temp0_15[0] + __expf((data1[(((((int)blockIdx.x) * 1000) + (k_outer * 128)) + ((int)threadIdx_x_0))] - T_softmax_maxelem4[0])));
        }
    }
    float red_buf0_215[1];
    unsigned int mask_216[1];
    float t0_217[1];
    float red_buf0_318[1];
    unsigned int mask_319[1];
    float t0_320[1];
    static float red_buf_staging_121[4] __attribute__((shared));
    red_buf0_318[0] = normal_reduce_temp0_15[0];
    mask_319[0] = __activemask();
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 16, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 8, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 4, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 2, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 1, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    if ((((int)threadIdx_x_0) % 32) == 0) {
        red_buf_staging_121[(((int)threadIdx_x_0) >> 5)] = red_buf0_318[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) < 4) {
        red_buf0_215[0] = red_buf_staging_121[((int)threadIdx_x_0)];
    }
    mask_216[0] = (__activemask() & (unsigned int)15);
    t0_217[0] = __shfl_down_sync(mask_216[0], red_buf0_215[0], 2, 32);
    red_buf0_215[0] = (red_buf0_215[0] + t0_217[0]);
    t0_217[0] = __shfl_down_sync(mask_216[0], red_buf0_215[0], 1, 32);
    red_buf0_215[0] = (red_buf0_215[0] + t0_217[0]);
    if (((int)threadIdx_x_0) == 0) {
        ((volatile float *)red_result_16)[0] = red_buf0_215[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) == 0) {
        T_softmax_expsum7[0] = ((volatile float *)red_result_16)[0];
    }
    asm ("bar.sync 1,128;");
    ;
    for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
        if (((i2_outer * 16) + (((int)threadIdx_x_0) >> 3)) < 125) {
            T_softmax_norm0[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_0))] = (__expf((data1[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_0))] - T_softmax_maxelem4[0])) / T_softmax_expsum7[0]);
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float normal_reduce_temp024[1];
    static float red_result25[257] __attribute__((shared));
    static float T_softmax_maxelem26[257] __attribute__((shared));
    float normal_reduce_temp0_127[1];
    static float red_result_128[257] __attribute__((shared));
    static float T_softmax_expsum29[257] __attribute__((shared));
    normal_reduce_temp024[0] = -3.40282306E+38F;
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1))]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 128)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 256)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 384)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 512)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 640)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 768)]);
    if (((int)threadIdx_x_1) < 104) {
        normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 896)]);
    }
    float red_buf030[1];
    unsigned int mask31[1];
    float t032[1];
    float red_buf0_133[1];
    unsigned int mask_134[1];
    float t0_135[1];
    static float red_buf_staging36[4] __attribute__((shared));
    red_buf0_133[0] = normal_reduce_temp024[0];
    mask_134[0] = __activemask();
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 16, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 8, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 4, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 2, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 1, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    if ((((int)threadIdx_x_1) % 32) == 0) {
        red_buf_staging36[(((int)threadIdx_x_1) >> 5)] = red_buf0_133[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) < 4) {
        red_buf030[0] = red_buf_staging36[((int)threadIdx_x_1)];
    }
    mask31[0] = (__activemask() & (unsigned int)15);
    t032[0] = __shfl_down_sync(mask31[0], red_buf030[0], 2, 32);
    red_buf030[0] = max(red_buf030[0], t032[0]);
    t032[0] = __shfl_down_sync(mask31[0], red_buf030[0], 1, 32);
    red_buf030[0] = max(red_buf030[0], t032[0]);
    if (((int)threadIdx_x_1) == 0) {
        ((volatile float *)red_result25)[0] = red_buf030[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) == 0) {
        T_softmax_maxelem26[0] = ((volatile float *)red_result25)[0];
    }
    normal_reduce_temp0_127[0] = 0.F;
    asm ("bar.sync 2,128;");
    ;
    for (int k_outer = 0; k_outer < 8; ++k_outer) {
        if (((k_outer * 16) + (((int)threadIdx_x_1) >> 3)) < 125) {
            normal_reduce_temp0_127[0] = (normal_reduce_temp0_127[0] + __expf((data23[(((((int)blockIdx.x) * 1000) + (k_outer * 128)) + ((int)threadIdx_x_1))] - T_softmax_maxelem26[0])));
        }
    }
    float red_buf0_237[1];
    unsigned int mask_238[1];
    float t0_239[1];
    float red_buf0_340[1];
    unsigned int mask_341[1];
    float t0_342[1];
    static float red_buf_staging_143[4] __attribute__((shared));
    red_buf0_340[0] = normal_reduce_temp0_127[0];
    mask_341[0] = __activemask();
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 16, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 8, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 4, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 2, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 1, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    if ((((int)threadIdx_x_1) % 32) == 0) {
        red_buf_staging_143[(((int)threadIdx_x_1) >> 5)] = red_buf0_340[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) < 4) {
        red_buf0_237[0] = red_buf_staging_143[((int)threadIdx_x_1)];
    }
    mask_238[0] = (__activemask() & (unsigned int)15);
    t0_239[0] = __shfl_down_sync(mask_238[0], red_buf0_237[0], 2, 32);
    red_buf0_237[0] = (red_buf0_237[0] + t0_239[0]);
    t0_239[0] = __shfl_down_sync(mask_238[0], red_buf0_237[0], 1, 32);
    red_buf0_237[0] = (red_buf0_237[0] + t0_239[0]);
    if (((int)threadIdx_x_1) == 0) {
        ((volatile float *)red_result_128)[0] = red_buf0_237[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) == 0) {
        T_softmax_expsum29[0] = ((volatile float *)red_result_128)[0];
    }
    asm ("bar.sync 2,128;");
    ;
    for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
        if (((i2_outer * 16) + (((int)threadIdx_x_1) >> 3)) < 125) {
            T_softmax_norm22[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_1))] = (__expf((data23[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_1))] - T_softmax_maxelem26[0])) / T_softmax_expsum29[0]);
        }
    }
}
}
 __global__ __launch_bounds__(128, 0) void softmax_shared_1036_softmax_shared_1036_copy_fused_kernel_vfuse_idx_0(float *__restrict T_softmax_norm0, float *__restrict data1, float *__restrict T_softmax_norm22, float *__restrict data23)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float normal_reduce_temp02[1];
    static float red_result3[257] __attribute__((shared));
    static float T_softmax_maxelem4[257] __attribute__((shared));
    float normal_reduce_temp0_15[1];
    static float red_result_16[257] __attribute__((shared));
    static float T_softmax_expsum7[257] __attribute__((shared));
    normal_reduce_temp02[0] = -3.40282306E+38F;
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0))]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 128)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 256)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 384)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 512)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 640)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 768)]);
    if (((int)threadIdx_x_0) < 104) {
        normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 896)]);
    }
    float red_buf08[1];
    unsigned int mask9[1];
    float t010[1];
    float red_buf0_111[1];
    unsigned int mask_112[1];
    float t0_113[1];
    static float red_buf_staging14[4] __attribute__((shared));
    red_buf0_111[0] = normal_reduce_temp02[0];
    mask_112[0] = __activemask();
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 16, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 8, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 4, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 2, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 1, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    if ((((int)threadIdx_x_0) % 32) == 0) {
        red_buf_staging14[(((int)threadIdx_x_0) >> 5)] = red_buf0_111[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) < 4) {
        red_buf08[0] = red_buf_staging14[((int)threadIdx_x_0)];
    }
    mask9[0] = (__activemask() & (unsigned int)15);
    t010[0] = __shfl_down_sync(mask9[0], red_buf08[0], 2, 32);
    red_buf08[0] = max(red_buf08[0], t010[0]);
    t010[0] = __shfl_down_sync(mask9[0], red_buf08[0], 1, 32);
    red_buf08[0] = max(red_buf08[0], t010[0]);
    if (((int)threadIdx_x_0) == 0) {
        ((volatile float *)red_result3)[0] = red_buf08[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) == 0) {
        T_softmax_maxelem4[0] = ((volatile float *)red_result3)[0];
    }
    normal_reduce_temp0_15[0] = 0.F;
    asm ("bar.sync 1,128;");
    ;
    for (int k_outer = 0; k_outer < 8; ++k_outer) {
        if (((k_outer * 16) + (((int)threadIdx_x_0) >> 3)) < 125) {
            normal_reduce_temp0_15[0] = (normal_reduce_temp0_15[0] + __expf((data1[(((((int)blockIdx.x) * 1000) + (k_outer * 128)) + ((int)threadIdx_x_0))] - T_softmax_maxelem4[0])));
        }
    }
    float red_buf0_215[1];
    unsigned int mask_216[1];
    float t0_217[1];
    float red_buf0_318[1];
    unsigned int mask_319[1];
    float t0_320[1];
    static float red_buf_staging_121[4] __attribute__((shared));
    red_buf0_318[0] = normal_reduce_temp0_15[0];
    mask_319[0] = __activemask();
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 16, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 8, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 4, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 2, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 1, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    if ((((int)threadIdx_x_0) % 32) == 0) {
        red_buf_staging_121[(((int)threadIdx_x_0) >> 5)] = red_buf0_318[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) < 4) {
        red_buf0_215[0] = red_buf_staging_121[((int)threadIdx_x_0)];
    }
    mask_216[0] = (__activemask() & (unsigned int)15);
    t0_217[0] = __shfl_down_sync(mask_216[0], red_buf0_215[0], 2, 32);
    red_buf0_215[0] = (red_buf0_215[0] + t0_217[0]);
    t0_217[0] = __shfl_down_sync(mask_216[0], red_buf0_215[0], 1, 32);
    red_buf0_215[0] = (red_buf0_215[0] + t0_217[0]);
    if (((int)threadIdx_x_0) == 0) {
        ((volatile float *)red_result_16)[0] = red_buf0_215[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) == 0) {
        T_softmax_expsum7[0] = ((volatile float *)red_result_16)[0];
    }
    asm ("bar.sync 1,128;");
    ;
    for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
        if (((i2_outer * 16) + (((int)threadIdx_x_0) >> 3)) < 125) {
            T_softmax_norm0[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_0))] = (__expf((data1[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_0))] - T_softmax_maxelem4[0])) / T_softmax_expsum7[0]);
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float normal_reduce_temp024[1];
    static float red_result25[257] __attribute__((shared));
    static float T_softmax_maxelem26[257] __attribute__((shared));
    float normal_reduce_temp0_127[1];
    static float red_result_128[257] __attribute__((shared));
    static float T_softmax_expsum29[257] __attribute__((shared));
    normal_reduce_temp024[0] = -3.40282306E+38F;
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1))]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 128)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 256)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 384)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 512)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 640)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 768)]);
    if (((int)threadIdx_x_1) < 104) {
        normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 896)]);
    }
    float red_buf030[1];
    unsigned int mask31[1];
    float t032[1];
    float red_buf0_133[1];
    unsigned int mask_134[1];
    float t0_135[1];
    static float red_buf_staging36[4] __attribute__((shared));
    red_buf0_133[0] = normal_reduce_temp024[0];
    mask_134[0] = __activemask();
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 16, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 8, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 4, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 2, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 1, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    if ((((int)threadIdx_x_1) % 32) == 0) {
        red_buf_staging36[(((int)threadIdx_x_1) >> 5)] = red_buf0_133[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) < 4) {
        red_buf030[0] = red_buf_staging36[((int)threadIdx_x_1)];
    }
    mask31[0] = (__activemask() & (unsigned int)15);
    t032[0] = __shfl_down_sync(mask31[0], red_buf030[0], 2, 32);
    red_buf030[0] = max(red_buf030[0], t032[0]);
    t032[0] = __shfl_down_sync(mask31[0], red_buf030[0], 1, 32);
    red_buf030[0] = max(red_buf030[0], t032[0]);
    if (((int)threadIdx_x_1) == 0) {
        ((volatile float *)red_result25)[0] = red_buf030[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) == 0) {
        T_softmax_maxelem26[0] = ((volatile float *)red_result25)[0];
    }
    normal_reduce_temp0_127[0] = 0.F;
    asm ("bar.sync 2,128;");
    ;
    for (int k_outer = 0; k_outer < 8; ++k_outer) {
        if (((k_outer * 16) + (((int)threadIdx_x_1) >> 3)) < 125) {
            normal_reduce_temp0_127[0] = (normal_reduce_temp0_127[0] + __expf((data23[(((((int)blockIdx.x) * 1000) + (k_outer * 128)) + ((int)threadIdx_x_1))] - T_softmax_maxelem26[0])));
        }
    }
    float red_buf0_237[1];
    unsigned int mask_238[1];
    float t0_239[1];
    float red_buf0_340[1];
    unsigned int mask_341[1];
    float t0_342[1];
    static float red_buf_staging_143[4] __attribute__((shared));
    red_buf0_340[0] = normal_reduce_temp0_127[0];
    mask_341[0] = __activemask();
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 16, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 8, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 4, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 2, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 1, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    if ((((int)threadIdx_x_1) % 32) == 0) {
        red_buf_staging_143[(((int)threadIdx_x_1) >> 5)] = red_buf0_340[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) < 4) {
        red_buf0_237[0] = red_buf_staging_143[((int)threadIdx_x_1)];
    }
    mask_238[0] = (__activemask() & (unsigned int)15);
    t0_239[0] = __shfl_down_sync(mask_238[0], red_buf0_237[0], 2, 32);
    red_buf0_237[0] = (red_buf0_237[0] + t0_239[0]);
    t0_239[0] = __shfl_down_sync(mask_238[0], red_buf0_237[0], 1, 32);
    red_buf0_237[0] = (red_buf0_237[0] + t0_239[0]);
    if (((int)threadIdx_x_1) == 0) {
        ((volatile float *)red_result_128)[0] = red_buf0_237[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) == 0) {
        T_softmax_expsum29[0] = ((volatile float *)red_result_128)[0];
    }
    asm ("bar.sync 2,128;");
    ;
    for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
        if (((i2_outer * 16) + (((int)threadIdx_x_1) >> 3)) < 125) {
            T_softmax_norm22[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_1))] = (__expf((data23[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_1))] - T_softmax_maxelem26[0])) / T_softmax_expsum29[0]);
        }
    }
}
}
 __global__ __launch_bounds__(256, 0) void softmax_shared_1036_softmax_shared_1036_copy_fused_kernel_hfuse_idx_0(float *__restrict T_softmax_norm0, float *__restrict data1, float *__restrict T_softmax_norm22, float *__restrict data23)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float normal_reduce_temp02[1];
    static float red_result3[257] __attribute__((shared));
    static float T_softmax_maxelem4[257] __attribute__((shared));
    float normal_reduce_temp0_15[1];
    static float red_result_16[257] __attribute__((shared));
    static float T_softmax_expsum7[257] __attribute__((shared));
    normal_reduce_temp02[0] = -3.40282306E+38F;
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0))]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 128)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 256)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 384)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 512)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 640)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 768)]);
    if (((int)threadIdx_x_0) < 104) {
        normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 896)]);
    }
    float red_buf08[1];
    unsigned int mask9[1];
    float t010[1];
    float red_buf0_111[1];
    unsigned int mask_112[1];
    float t0_113[1];
    static float red_buf_staging14[4] __attribute__((shared));
    red_buf0_111[0] = normal_reduce_temp02[0];
    mask_112[0] = __activemask();
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 16, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 8, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 4, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 2, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 1, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    if ((((int)threadIdx_x_0) % 32) == 0) {
        red_buf_staging14[(((int)threadIdx_x_0) >> 5)] = red_buf0_111[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) < 4) {
        red_buf08[0] = red_buf_staging14[((int)threadIdx_x_0)];
    }
    mask9[0] = (__activemask() & (unsigned int)15);
    t010[0] = __shfl_down_sync(mask9[0], red_buf08[0], 2, 32);
    red_buf08[0] = max(red_buf08[0], t010[0]);
    t010[0] = __shfl_down_sync(mask9[0], red_buf08[0], 1, 32);
    red_buf08[0] = max(red_buf08[0], t010[0]);
    if (((int)threadIdx_x_0) == 0) {
        ((volatile float *)red_result3)[0] = red_buf08[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) == 0) {
        T_softmax_maxelem4[0] = ((volatile float *)red_result3)[0];
    }
    normal_reduce_temp0_15[0] = 0.F;
    asm ("bar.sync 1,128;");
    ;
    for (int k_outer = 0; k_outer < 8; ++k_outer) {
        if (((k_outer * 16) + (((int)threadIdx_x_0) >> 3)) < 125) {
            normal_reduce_temp0_15[0] = (normal_reduce_temp0_15[0] + __expf((data1[(((((int)blockIdx.x) * 1000) + (k_outer * 128)) + ((int)threadIdx_x_0))] - T_softmax_maxelem4[0])));
        }
    }
    float red_buf0_215[1];
    unsigned int mask_216[1];
    float t0_217[1];
    float red_buf0_318[1];
    unsigned int mask_319[1];
    float t0_320[1];
    static float red_buf_staging_121[4] __attribute__((shared));
    red_buf0_318[0] = normal_reduce_temp0_15[0];
    mask_319[0] = __activemask();
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 16, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 8, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 4, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 2, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 1, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    if ((((int)threadIdx_x_0) % 32) == 0) {
        red_buf_staging_121[(((int)threadIdx_x_0) >> 5)] = red_buf0_318[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) < 4) {
        red_buf0_215[0] = red_buf_staging_121[((int)threadIdx_x_0)];
    }
    mask_216[0] = (__activemask() & (unsigned int)15);
    t0_217[0] = __shfl_down_sync(mask_216[0], red_buf0_215[0], 2, 32);
    red_buf0_215[0] = (red_buf0_215[0] + t0_217[0]);
    t0_217[0] = __shfl_down_sync(mask_216[0], red_buf0_215[0], 1, 32);
    red_buf0_215[0] = (red_buf0_215[0] + t0_217[0]);
    if (((int)threadIdx_x_0) == 0) {
        ((volatile float *)red_result_16)[0] = red_buf0_215[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) == 0) {
        T_softmax_expsum7[0] = ((volatile float *)red_result_16)[0];
    }
    asm ("bar.sync 1,128;");
    ;
    for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
        if (((i2_outer * 16) + (((int)threadIdx_x_0) >> 3)) < 125) {
            T_softmax_norm0[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_0))] = (__expf((data1[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_0))] - T_softmax_maxelem4[0])) / T_softmax_expsum7[0]);
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    float normal_reduce_temp024[1];
    static float red_result25[257] __attribute__((shared));
    static float T_softmax_maxelem26[257] __attribute__((shared));
    float normal_reduce_temp0_127[1];
    static float red_result_128[257] __attribute__((shared));
    static float T_softmax_expsum29[257] __attribute__((shared));
    normal_reduce_temp024[0] = -3.40282306E+38F;
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1))]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 128)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 256)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 384)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 512)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 640)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 768)]);
    if (((int)threadIdx_x_1) < 104) {
        normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 896)]);
    }
    float red_buf030[1];
    unsigned int mask31[1];
    float t032[1];
    float red_buf0_133[1];
    unsigned int mask_134[1];
    float t0_135[1];
    static float red_buf_staging36[4] __attribute__((shared));
    red_buf0_133[0] = normal_reduce_temp024[0];
    mask_134[0] = __activemask();
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 16, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 8, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 4, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 2, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 1, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    if ((((int)threadIdx_x_1) % 32) == 0) {
        red_buf_staging36[(((int)threadIdx_x_1) >> 5)] = red_buf0_133[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) < 4) {
        red_buf030[0] = red_buf_staging36[((int)threadIdx_x_1)];
    }
    mask31[0] = (__activemask() & (unsigned int)15);
    t032[0] = __shfl_down_sync(mask31[0], red_buf030[0], 2, 32);
    red_buf030[0] = max(red_buf030[0], t032[0]);
    t032[0] = __shfl_down_sync(mask31[0], red_buf030[0], 1, 32);
    red_buf030[0] = max(red_buf030[0], t032[0]);
    if (((int)threadIdx_x_1) == 0) {
        ((volatile float *)red_result25)[0] = red_buf030[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) == 0) {
        T_softmax_maxelem26[0] = ((volatile float *)red_result25)[0];
    }
    normal_reduce_temp0_127[0] = 0.F;
    asm ("bar.sync 2,128;");
    ;
    for (int k_outer = 0; k_outer < 8; ++k_outer) {
        if (((k_outer * 16) + (((int)threadIdx_x_1) >> 3)) < 125) {
            normal_reduce_temp0_127[0] = (normal_reduce_temp0_127[0] + __expf((data23[(((((int)blockIdx.x) * 1000) + (k_outer * 128)) + ((int)threadIdx_x_1))] - T_softmax_maxelem26[0])));
        }
    }
    float red_buf0_237[1];
    unsigned int mask_238[1];
    float t0_239[1];
    float red_buf0_340[1];
    unsigned int mask_341[1];
    float t0_342[1];
    static float red_buf_staging_143[4] __attribute__((shared));
    red_buf0_340[0] = normal_reduce_temp0_127[0];
    mask_341[0] = __activemask();
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 16, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 8, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 4, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 2, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 1, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    if ((((int)threadIdx_x_1) % 32) == 0) {
        red_buf_staging_143[(((int)threadIdx_x_1) >> 5)] = red_buf0_340[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) < 4) {
        red_buf0_237[0] = red_buf_staging_143[((int)threadIdx_x_1)];
    }
    mask_238[0] = (__activemask() & (unsigned int)15);
    t0_239[0] = __shfl_down_sync(mask_238[0], red_buf0_237[0], 2, 32);
    red_buf0_237[0] = (red_buf0_237[0] + t0_239[0]);
    t0_239[0] = __shfl_down_sync(mask_238[0], red_buf0_237[0], 1, 32);
    red_buf0_237[0] = (red_buf0_237[0] + t0_239[0]);
    if (((int)threadIdx_x_1) == 0) {
        ((volatile float *)red_result_128)[0] = red_buf0_237[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) == 0) {
        T_softmax_expsum29[0] = ((volatile float *)red_result_128)[0];
    }
    asm ("bar.sync 2,128;");
    ;
    for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
        if (((i2_outer * 16) + (((int)threadIdx_x_1) >> 3)) < 125) {
            T_softmax_norm22[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_1))] = (__expf((data23[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_1))] - T_softmax_maxelem26[0])) / T_softmax_expsum29[0]);
        }
    }
}
}
 __global__ __launch_bounds__(256, 8) void softmax_shared_1036_softmax_shared_1036_copy_fused_kernel_hfuse_lb_idx_0(float *__restrict T_softmax_norm0, float *__restrict data1, float *__restrict T_softmax_norm22, float *__restrict data23)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    float normal_reduce_temp02[1];
    static float red_result3[257] __attribute__((shared));
    static float T_softmax_maxelem4[257] __attribute__((shared));
    float normal_reduce_temp0_15[1];
    static float red_result_16[257] __attribute__((shared));
    static float T_softmax_expsum7[257] __attribute__((shared));
    normal_reduce_temp02[0] = -3.40282306E+38F;
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0))]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 128)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 256)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 384)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 512)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 640)]);
    normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 768)]);
    if (((int)threadIdx_x_0) < 104) {
        normal_reduce_temp02[0] = max(normal_reduce_temp02[0], data1[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_0)) + 896)]);
    }
    float red_buf08[1];
    unsigned int mask9[1];
    float t010[1];
    float red_buf0_111[1];
    unsigned int mask_112[1];
    float t0_113[1];
    static float red_buf_staging14[4] __attribute__((shared));
    red_buf0_111[0] = normal_reduce_temp02[0];
    mask_112[0] = __activemask();
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 16, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 8, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 4, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 2, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    t0_113[0] = __shfl_down_sync(mask_112[0], red_buf0_111[0], 1, 32);
    red_buf0_111[0] = max(red_buf0_111[0], t0_113[0]);
    if ((((int)threadIdx_x_0) % 32) == 0) {
        red_buf_staging14[(((int)threadIdx_x_0) >> 5)] = red_buf0_111[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) < 4) {
        red_buf08[0] = red_buf_staging14[((int)threadIdx_x_0)];
    }
    mask9[0] = (__activemask() & (unsigned int)15);
    t010[0] = __shfl_down_sync(mask9[0], red_buf08[0], 2, 32);
    red_buf08[0] = max(red_buf08[0], t010[0]);
    t010[0] = __shfl_down_sync(mask9[0], red_buf08[0], 1, 32);
    red_buf08[0] = max(red_buf08[0], t010[0]);
    if (((int)threadIdx_x_0) == 0) {
        ((volatile float *)red_result3)[0] = red_buf08[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) == 0) {
        T_softmax_maxelem4[0] = ((volatile float *)red_result3)[0];
    }
    normal_reduce_temp0_15[0] = 0.F;
    asm ("bar.sync 1,128;");
    ;
    for (int k_outer = 0; k_outer < 8; ++k_outer) {
        if (((k_outer * 16) + (((int)threadIdx_x_0) >> 3)) < 125) {
            normal_reduce_temp0_15[0] = (normal_reduce_temp0_15[0] + __expf((data1[(((((int)blockIdx.x) * 1000) + (k_outer * 128)) + ((int)threadIdx_x_0))] - T_softmax_maxelem4[0])));
        }
    }
    float red_buf0_215[1];
    unsigned int mask_216[1];
    float t0_217[1];
    float red_buf0_318[1];
    unsigned int mask_319[1];
    float t0_320[1];
    static float red_buf_staging_121[4] __attribute__((shared));
    red_buf0_318[0] = normal_reduce_temp0_15[0];
    mask_319[0] = __activemask();
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 16, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 8, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 4, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 2, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    t0_320[0] = __shfl_down_sync(mask_319[0], red_buf0_318[0], 1, 32);
    red_buf0_318[0] = (red_buf0_318[0] + t0_320[0]);
    if ((((int)threadIdx_x_0) % 32) == 0) {
        red_buf_staging_121[(((int)threadIdx_x_0) >> 5)] = red_buf0_318[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) < 4) {
        red_buf0_215[0] = red_buf_staging_121[((int)threadIdx_x_0)];
    }
    mask_216[0] = (__activemask() & (unsigned int)15);
    t0_217[0] = __shfl_down_sync(mask_216[0], red_buf0_215[0], 2, 32);
    red_buf0_215[0] = (red_buf0_215[0] + t0_217[0]);
    t0_217[0] = __shfl_down_sync(mask_216[0], red_buf0_215[0], 1, 32);
    red_buf0_215[0] = (red_buf0_215[0] + t0_217[0]);
    if (((int)threadIdx_x_0) == 0) {
        ((volatile float *)red_result_16)[0] = red_buf0_215[0];
    }
    asm ("bar.sync 1,128;");
    ;
    if (((int)threadIdx_x_0) == 0) {
        T_softmax_expsum7[0] = ((volatile float *)red_result_16)[0];
    }
    asm ("bar.sync 1,128;");
    ;
    for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
        if (((i2_outer * 16) + (((int)threadIdx_x_0) >> 3)) < 125) {
            T_softmax_norm0[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_0))] = (__expf((data1[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_0))] - T_softmax_maxelem4[0])) / T_softmax_expsum7[0]);
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    float normal_reduce_temp024[1];
    static float red_result25[257] __attribute__((shared));
    static float T_softmax_maxelem26[257] __attribute__((shared));
    float normal_reduce_temp0_127[1];
    static float red_result_128[257] __attribute__((shared));
    static float T_softmax_expsum29[257] __attribute__((shared));
    normal_reduce_temp024[0] = -3.40282306E+38F;
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1))]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 128)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 256)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 384)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 512)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 640)]);
    normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 768)]);
    if (((int)threadIdx_x_1) < 104) {
        normal_reduce_temp024[0] = max(normal_reduce_temp024[0], data23[(((((int)blockIdx.x) * 1000) + ((int)threadIdx_x_1)) + 896)]);
    }
    float red_buf030[1];
    unsigned int mask31[1];
    float t032[1];
    float red_buf0_133[1];
    unsigned int mask_134[1];
    float t0_135[1];
    static float red_buf_staging36[4] __attribute__((shared));
    red_buf0_133[0] = normal_reduce_temp024[0];
    mask_134[0] = __activemask();
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 16, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 8, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 4, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 2, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    t0_135[0] = __shfl_down_sync(mask_134[0], red_buf0_133[0], 1, 32);
    red_buf0_133[0] = max(red_buf0_133[0], t0_135[0]);
    if ((((int)threadIdx_x_1) % 32) == 0) {
        red_buf_staging36[(((int)threadIdx_x_1) >> 5)] = red_buf0_133[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) < 4) {
        red_buf030[0] = red_buf_staging36[((int)threadIdx_x_1)];
    }
    mask31[0] = (__activemask() & (unsigned int)15);
    t032[0] = __shfl_down_sync(mask31[0], red_buf030[0], 2, 32);
    red_buf030[0] = max(red_buf030[0], t032[0]);
    t032[0] = __shfl_down_sync(mask31[0], red_buf030[0], 1, 32);
    red_buf030[0] = max(red_buf030[0], t032[0]);
    if (((int)threadIdx_x_1) == 0) {
        ((volatile float *)red_result25)[0] = red_buf030[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) == 0) {
        T_softmax_maxelem26[0] = ((volatile float *)red_result25)[0];
    }
    normal_reduce_temp0_127[0] = 0.F;
    asm ("bar.sync 2,128;");
    ;
    for (int k_outer = 0; k_outer < 8; ++k_outer) {
        if (((k_outer * 16) + (((int)threadIdx_x_1) >> 3)) < 125) {
            normal_reduce_temp0_127[0] = (normal_reduce_temp0_127[0] + __expf((data23[(((((int)blockIdx.x) * 1000) + (k_outer * 128)) + ((int)threadIdx_x_1))] - T_softmax_maxelem26[0])));
        }
    }
    float red_buf0_237[1];
    unsigned int mask_238[1];
    float t0_239[1];
    float red_buf0_340[1];
    unsigned int mask_341[1];
    float t0_342[1];
    static float red_buf_staging_143[4] __attribute__((shared));
    red_buf0_340[0] = normal_reduce_temp0_127[0];
    mask_341[0] = __activemask();
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 16, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 8, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 4, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 2, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    t0_342[0] = __shfl_down_sync(mask_341[0], red_buf0_340[0], 1, 32);
    red_buf0_340[0] = (red_buf0_340[0] + t0_342[0]);
    if ((((int)threadIdx_x_1) % 32) == 0) {
        red_buf_staging_143[(((int)threadIdx_x_1) >> 5)] = red_buf0_340[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) < 4) {
        red_buf0_237[0] = red_buf_staging_143[((int)threadIdx_x_1)];
    }
    mask_238[0] = (__activemask() & (unsigned int)15);
    t0_239[0] = __shfl_down_sync(mask_238[0], red_buf0_237[0], 2, 32);
    red_buf0_237[0] = (red_buf0_237[0] + t0_239[0]);
    t0_239[0] = __shfl_down_sync(mask_238[0], red_buf0_237[0], 1, 32);
    red_buf0_237[0] = (red_buf0_237[0] + t0_239[0]);
    if (((int)threadIdx_x_1) == 0) {
        ((volatile float *)red_result_128)[0] = red_buf0_237[0];
    }
    asm ("bar.sync 2,128;");
    ;
    if (((int)threadIdx_x_1) == 0) {
        T_softmax_expsum29[0] = ((volatile float *)red_result_128)[0];
    }
    asm ("bar.sync 2,128;");
    ;
    for (int i2_outer = 0; i2_outer < 8; ++i2_outer) {
        if (((i2_outer * 16) + (((int)threadIdx_x_1) >> 3)) < 125) {
            T_softmax_norm22[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_1))] = (__expf((data23[(((((int)blockIdx.x) * 1000) + (i2_outer * 128)) + ((int)threadIdx_x_1))] - T_softmax_maxelem26[0])) / T_softmax_expsum29[0]);
        }
    }
}
}
